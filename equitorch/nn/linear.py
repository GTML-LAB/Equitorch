from math import sqrt
from typing import Optional, Union
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.utils import scatter, segment

from ..math import dot


from ..typing import DegreeRange

from ..utils.indices import (
    degree_order_to_index,
    expand_degree_to_order,
    check_degree_range,
    extract_batch_ptr,
    extract_in_degree,
    num_orders_in,
    degrees_in_range,
    pad_to_degree
)
from ..utils.clebsch_gordan import coo_CG

class DegreeWiseLinear(nn.Module):
    r"""
    The degree-wise linear (channel mixing) operation.  

    This class implements the self-interaction operation described in `Tensor field 
    networks: Rotation- and translation-equivariant neural networks for 3D
    point clouds <https://arxiv.org/abs/1802.08219>`_.

    .. math::
        \begin{aligned}
        \mathbf{x'}_c &= \bigoplus_{l\in L_{\text{out}}}\mathbf{x'}^{(l)}_c,\\
        \mathbf{x'}^{(l)}_c &=\begin{cases}
            \sum_{c'} \mathbf{W}^{(l)}_{cc'} \mathbf{x}_{c'}^{(l)}, & l \in L_{\text{in}},\\
            \mathbf{0}, & l \notin L_{\text{in}}.
            \end{cases}
        \end{aligned}

    Note
    ----
    The degrees in :obj:`L_out` but not in :obj:`L_in` will be padded with zeros, and the degrees in
    :obj:`L_in` but not in :obj:`L_out` will be ignored.

    Parameters
    ----------
    L_in : :obj:`~equitorch.typing.DegreeRange`
        Input degree range.
    L_out : :obj:`~equitorch.typing.DegreeRange`
        Output degree range.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.

    """

    def __init__(self, 
                 L_in: DegreeRange,
                 L_out: DegreeRange,
                 in_channels: int, 
                 out_channels: int,
                ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.L_in = check_degree_range(L_in)
        self.L_out = check_degree_range(L_out)
        self.L_eff = (max(self.L_in[0], self.L_out[0]), min(self.L_in[1],self.L_out[1]))
        assert self.L_eff[0] <= self.L_eff[1], "The input and output degree range should have intersections!"
        self.weight = self.self_interaction_weight = nn.Parameter(
                    torch.randn(self.L_eff[1]+1 - self.L_eff[0], in_channels, out_channels) * 2 / sqrt(in_channels + out_channels))
        self.L_in
    def forward(self, x: Tensor):
        r"""
        Parameters
        ----------
        x : :obj:`~torch.Tensor`
            The input tensor of shape :math:`(N, \text{num_orders_in}, C_{\text{in}})`, 
            where :math:`N` is the batch size and :math:`C_{\text{in}}` is the number of channels.

        Returns
        -------
        :obj:`~torch.Tensor`
            The output tensor of shape :math:`(N, \text{num_orders_out}, C_{\text{out}})`.
        """
        x = extract_in_degree(x, self.L_in, self.L_eff)
        out = (x.unsqueeze(-2) @ 
                expand_degree_to_order(self.weight.unsqueeze(0), self.L_eff, dim=-3)).squeeze(-2)
        out = pad_to_degree(out, self.L_eff, self.L_out)
        return out
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(L_in={self.L_in}, L_out={self.L_out}, in_channels={self.in_channels}, out_channels={self.out_channels})'

def _so3_conv(feature: Tensor, 
              sh: Tensor,
              weight: Tensor,
              CG_vals: Tensor,
              Ml1l2_ptr: Tensor, 
              l_ind: Tensor, 
              M1: Tensor,
              M2: Tensor,
              M_ptr: Tensor, 
              ):
    '''
        feature: B * M1 * C_i
        weight: B * ll1l2 * C_i * C_o
    '''
    feature = feature.index_select(-2, M1) # B * MM1M2 * C
    sh = sh.index_select(-1, M2) # B * MM1M2
    # C_val: MM1M2
    inter = (CG_vals.unsqueeze(0) * sh).unsqueeze(-1) * feature # B * MM1M2 * C
    inter = segment(inter, Ml1l2_ptr.unsqueeze(0))  # B * Ml1l2 * C
    weight = weight.index_select(dim=1,index=l_ind) # B * Ml1l2 * C (* C')

    ret = (inter.unsqueeze(-2) @ weight).squeeze(-2) # B * Ml1l2 * C'

    ret = segment(ret, M_ptr.unsqueeze(0)) # B * M * C'
    return ret

def _so3_cw_conv(feature: Tensor, 
              sh: Tensor,
              weight: Tensor,
              CG_vals: Tensor,
              M_ptr: Tensor, 
              l_ind: Tensor, 
              M1: Tensor,
              M2: Tensor,
              ):
    '''
        feature: B * M1 * C
        weight: B * ll1l2 * C
    '''
    feature = feature.index_select(-2, M1) # B * MM1M2 * C
    sh = sh.index_select(-1, M2) # B * MM1M2
    # C_val: MM1M2
    inter = (CG_vals.unsqueeze(0) * sh).unsqueeze(-1) * feature # B * MM1M2 * C
    weight = weight.index_select(dim=1,index=l_ind) # B * MM1M2 * C
    ret = inter * weight # B * MM1M2 * C'
    ret = segment(ret, M_ptr.unsqueeze(0)) # B * M * C
    return ret


class SO3Linear(nn.Module):
    r"""
    The SO(3) equivariant linear operation.

    The SO(3) equivariant linear operation first proposed by `Tensor field 
    networks: Rotation- and translation-equivariant neural networks for 3D
    point clouds <https://arxiv.org/abs/1802.08219>`_. The name of "linear 
    operation" comes from `Geometric and Physical Quantities Improve E(3) 
    Equivariant Message Passing <https://arxiv.org/abs/2110.02905>`_.

    This operation can be expressed as
    
    .. math::
        \mathbf{x'}^{(l)}=\sum_{l_1,l_2}\mathbf{W}_{l_1,l_2}^{l}\mathbf{x}^{(l_1)}\otimes \mathbf{Y}^{(l_2)}(\mathbf{r}),
    
    or

    .. math::
        \mathbf{x'}^{(l)}_{m}=\sum_{l_1,l_2}\sum_{m_1,m_2}C_{(l_1,m_1)(l_2,m_2)}^{(l,m)}\mathbf{W}_{l_1,l_2}^{l}\mathbf{x}_{m_1}^{(l_1)}\mathbf{Y }_{m_2}^{(l_2)}(\mathbf{r}),
    
    where the summation of :math:`(l_1,l_2)` is over all the values such that 
    :math:`l_1\in L_1, l_2\in L_2` and :math:`|l_1-l_2|\le l\le l_1+l_2`,
    :math:`C_{(l_1,m_1)(l_2,m_2)}^{(l,m)}` are the Clebsch-Gordan coefficients and 
    :math:`\mathbf{W}_{l_1,l_2}^{l}\mathbf{x}_{m_1}^{(l_i)}` means 
    :math:`\sum_{c'}\mathbf{W}_{l_1,l_2,cc'}^{l}\mathbf{x}_{m_1,c'}^{(l_i)}`
    if :obj:`channel_wise` is :obj:`False`, or
    :math:`\mathbf{W}_{l_1,l_2,c}^{l}\mathbf{x}_{m_1,c}^{(l_i)}` if :obj:`channel_wise` is :obj:`True`.

    When there are no ambiguities on :obj:`L_in`, :obj:`L_edge` and :obj:`L_out`,
    we also denote this operation as
    
    .. math::
        \mathbf{x'}=\tilde{\mathbf{W}}({\mathbf{r}})\mathbf{x},

    We use "tilde" to denote this is an equivariant operation. 
    This looks more like a linear operation where the weight can depend on :math:`\mathbf{r}`.

    The SO(3) equivariance means that for any rotation matrix :math:`\mathbf{R}\in\mathrm{SO(3)}`
    and corresponding Wigner D matrices :math:`\mathbf{D}_{\text{in}}`, 
    :math:`\mathbf{D}_{\text{out}}` in input/output feature spaces, it satisfies that

    .. math::
        \mathbf{D}_{\text{out}}\tilde{\mathbf{W}}({\mathbf{r}})\mathbf{x}=\tilde{\mathbf{W}}(\mathbf{R}{\mathbf{r}})(\mathbf{D}_{\text{in}}\mathbf{x}).

    .. note::
        By using sparse contraction on :math:`m_1,m_2`, the time complexity of this operation is :math:`O(L^5)`
        for the maximum degree :math:`L`.

        Whenever possible, it is recommended to use :obj:`SO2Linear` for 
        equivariant operation on large :math:`L`.

    Parameters
    ----------
    L_in : :obj:`~equitorch.typing.DegreeRange`
        The degree range of the input.
    L_edge : :obj:`~equitorch.typing.DegreeRange`
        The degree range of the edge.
    L_out : :obj:`~equitorch.typing.DegreeRange`
        The degree range of the output.
    in_channels : int
        The number of input channels.
    out_channels : int
        The number of output channels.
    external_weights : bool, optional
        Whether to use an external weights. Defaults to False.
    channel_wise : bool, optional
        Whether to perform the operation channel-wise. Defaults to False.
    """
    def __init__(
            self,
            L_in: DegreeRange,
            L_edge: DegreeRange,
            L_out: DegreeRange,
            in_channels: int,
            out_channels: int,
            external_weights: bool = False,
            channel_wise: bool = False,
    ):
        super().__init__()

        assert in_channels == out_channels or not channel_wise

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.L_in = check_degree_range(L_in)
        self.L_out = check_degree_range(L_out)
        self.L_edge = check_degree_range(L_edge)
        self.external_weights = external_weights
        
        self.channel_wise = channel_wise

        CG_vals, Ms, ls_cg, l_ind_cg = coo_CG(L_out, L_in, L_edge)
        
        if self.channel_wise:
            M_batch, M_ptr, M = extract_batch_ptr([Ms[0]])

            l_ind = l_ind_cg

            self.register_buffer('Ms', Ms)
            self.register_buffer('M1', Ms[1])
            self.register_buffer('M2', Ms[2])
            self.register_buffer('l_ind', l_ind)
            self.register_buffer('M_ptr', M_ptr)
            self.register_buffer('CG_vals', CG_vals)

            self.num_weights = l_ind.unique().numel()

            self.weight_shape = (self.num_weights, self.in_channels)

            if not self.external_weights:
                self.weight = nn.Parameter(
                    torch.randn(1, self.num_weights,
                                self.in_channels))
        else:
            Ml1l2_batch, Ml1l2_ptr, Ml1l2 = extract_batch_ptr([Ms[0], ls_cg[1], ls_cg[2]])
            
            l_ind = l_ind_cg[Ml1l2_ptr[:-1]]
            M_batch, M_ptr, M = extract_batch_ptr(Ml1l2[:1])

            self.register_buffer('Ml1l2_ptr', Ml1l2_ptr)
            self.register_buffer('l_ind', l_ind)
            self.register_buffer('M1', Ms[1])
            self.register_buffer('M2', Ms[2])
            self.register_buffer('M_ptr', M_ptr)
            self.register_buffer('CG_vals', CG_vals)

            self.num_weights = l_ind.unique().numel()

            self.weight_shape = (self.num_weights, self.in_channels, self.out_channels)

            if not self.external_weights:
                self.weight = nn.Parameter(
                    torch.randn(1, self.num_weights,
                                in_channels, out_channels) \
                                    * sqrt(2/(in_channels+out_channels)))
            
    def forward(self, 
                x: Tensor, 
                sh: Tensor,
                weight: Optional[Tensor] = None,
        ):
        r"""
        Applies the SO(3) linear operation to the input tensor.

        Parameters
        ----------
        x : :obj:`~torch.Tensor`
            The input tensor of shape :math:`(N, \text{num_orders_in}, C_{\text{in}})`, 
            where :math:`N` is the batch size and :math:`C_{\text{in}}` is the number of channels.
        sh : :obj:`~torch.Tensor`
            The edge spherical harmonics tensor of shape :math:`(N, \text{num_orders_edge})`.
        weight : :obj:`~torch.Tensor`, optional
            The external weights to use for the linear operation. If :obj:`None`, the 
            internal weights will be used.
            The shape of the weights depends on the value of :obj:`channel_wise`. 
            If :obj:`channel_wise` is :obj:`True`, the shape should be :math:`(N, \text{num_weights}, C_{\text{in}})`.
            If :obj:`channel_wise` is :obj:`False`, the shape should be :math:`(N, \text{num_weights}, C_{\text{in}}, C_{\text{out}})`, 

        Returns
        -------
        :obj:`~torch.Tensor`
            The output tensor of shape :math:`(N, \text{num_orders_out}, C_{\text{out}})`.

        Note
        ----
        If :obj:`external_weights` is :obj:`True`, the :obj:`weight` parameter must be provided. 
        If :obj:`external_weights` is :obj:`False`, the :obj:`weight` will still be used if provided.
        """
        if weight is None and not self.external_weights:
            weight = self.weight
        else:
            weight = weight.view(-1, *(self.weight_shape))

        if self.channel_wise:
            return _so3_cw_conv(x, sh, weight,
                         self.CG_vals, self.M_ptr, self.l_ind, 
                         self.M1, self.M2)
        else:
            return _so3_conv(x, sh, weight,
                            self.CG_vals, self.Ml1l2_ptr, self.l_ind, 
                            self.M1, self.M2, self.M_ptr)
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(\n  L_in={self.L_in}, L_edge={self.L_edge}, L_out={self.L_out},\n  in_channels={self.in_channels}, out_channels={self.out_channels}, channel_wise={self.channel_wise}, external_weights={self.external_weights}\n)'


def _so2_indices(L_in: DegreeRange, L_out: DegreeRange):
    L_in = check_degree_range(L_in)
    L_out = check_degree_range(L_out)
    ret = sorted([
        (degree_order_to_index(l_out, m_out, L_out[0]), 
         degree_order_to_index(l_in, m_in, L_in[0]),
         l_in,
         l_out,
         -1. if m_out > 0 and m_in < 0 else 1.,
         weight_idx,
         )
        # (weight_idx)
        for weight_idx, (l_out, l_in, m_weight) in enumerate(
            (l_out, l_in, m_weight) 
            for l_out in degrees_in_range(L_out)
            for l_in in degrees_in_range(L_in)
            for m_weight in range(-min(l_out, l_in), min(l_out, l_in)+1)
        ) 
        for m_out, m_in in (((-abs(m_weight),-m_weight), (abs(m_weight),m_weight)) if m_weight != 0 else ((0,0),))
    ])
    Ms = torch.tensor([[t[0], t[1]] for t in ret]).T
    ls = torch.tensor([[t[2], t[3]] for t in ret]).T
    weight_sign = torch.tensor([t[4] for t in ret])
    weight_idx = torch.tensor([t[5] for t in ret])
    return Ms, ls, weight_sign, weight_idx

class SO2Linear(nn.Module):

    r'''
    The SO(2) equivariant linear operation as described in the paper `Reducing SO(3)
    Convolutions to SO(2) for Efficient Equivariant GNNs 
    <https://arxiv.org/abs/2302.03655>`_.

    .. math::
        \begin{aligned}
        \mathbf{x'}_{m}^{(l_o)}&=\sum_{l_i\in L_{in}}\mathbf{W}_{m}^{(l_o,l_i)}\mathbf{x}_{-m}^{(l_i)}+\mathbf{W}_{-m}^{(l_o,l_i)}\mathbf{x}_{m}^{(l_i)}, & m < 0,\\
        \mathbf{x'}_{0}^{(l_o)}&=\sum_{l_i\in L_{in}}\mathbf{W}_{0}^{(l_o,l_i)}\mathbf{x}_{0}^{(l_i)}, &\\
        \mathbf{x'}_{m}^{(l_o)}&=\sum_{l_i\in L_{in}}\mathbf{W}_{m}^{(l_o,l_i)}\mathbf{x}_{m}^{(l_i)}-\mathbf{W}_{-m}^{(l_o,l_i)}\mathbf{x}_{-m}^{(l_i)}, & m > 0,\\
        \end{aligned}

    where :math:`\mathbf{W}_{m}^{(l_o,l_i)}\mathbf{x}_{m'}^{(l_i)}` means 
    :math:`\sum_{c'}\mathbf{W}_{m,cc'}^{(l_o,l_i)}\mathbf{x}_{m',c'}^{(l_i)}`
    if :obj:`channel_wise` is :obj:`False`, or
    :math:`\mathbf{W}_{m,c}^{(l_o,l_i)}\mathbf{x}_{m',c}^{(l_i)}` if :obj:`channel_wise` is :obj:`True`.

    When there is no ambiguity, we also denote the operation as 

    .. math::
        \mathbf{x}'=\tilde{\mathbf{W}}_{\phi}\mathbf{x} \text{ or } \mathbf{x}'=\tilde{\mathbf{W}}_{\text{SO(2)}}\mathbf{x}.


    The SO(2) equivariance means that, for any rotation 
    :math:`\mathbf{R}=\begin{bmatrix}\cos\phi&-\sin\phi&0\\\sin\phi&\cos\phi&0\\0&0&1\end{bmatrix}`
    around z-axis and corresponding Wigner D matrices :math:`\mathbf{D}_{\text{in}}`, 
    :math:`\mathbf{D}_{\text{out}}` in input/output feature spaces,
    it satisfise that

    .. math::
        \mathbf{D}_{\text{out}}\tilde{\mathbf{W}}_{\phi}\mathbf{x}=\tilde{\mathbf{W}}_{\phi}(\mathbf{D}_{\text{in}}\mathbf{x}).


    This operation can work as a more efficient alternative for SO(3) linear operation
    and satisfies: 
    
    for any possible SO(3) linear operation :math:`\tilde{\mathbf{W}}(\mathbf{r})`,
    there exists an SO(2) linear operation :math:`\tilde{\mathbf{W}}_{\phi}'` such that:

    .. math::
        \mathbf{D}_{\mathbf{r},\text{out}}^\top\tilde{\mathbf{W}}'_{\phi}(\mathbf{D}_{\mathbf{r},\text{in}}\mathbf{x})=\tilde{\mathbf{W}}(\mathbf {r})\mathbf{x}

    and vice versa, where :math:`\mathbf{D}_{\mathbf{r},\text{in}}` and :math:`\mathbf{D}_{\mathbf{r},\text{out}}` are the Wigner D
    matrices on the input/output spaces corresponding to the rotation matrix that can align :math:`\mathbf{r}`
    to the z axis. (See Appendix 2 of the paper above for the proof of the bijection.)

    To explicitly convert from or to the weight for :obj:`SO3Linear` operation, see 
    :obj:`~equitorch.utils.so3_weights_to_so2` and :obj:`~equitorch.utils.so2_weights_to_so3`.
    
    .. note::
        When channels :math:`C` explicitly considered, the linear operation will 
        be of complexity :math:`O(L^3C)` if channel wise or :math:`O(L^3CC')` if
        not, but the rotation :math:`\mathbf{Dx}` will always be of complexity
        :math:`O(L^4C)` if dense Wigner D matrices are used. However, since the transform
        :math:`\mathbf{Dx}` can be performed via a dense matrix multiplication,
        this will not be likely to be the bottleneck in practice.

        Whenever possible, it is recommended to use :obj:`SO2Linear` rather than :obj:`SO3Linear` for 
        equivariant operation on large :math:`L`.

    Parameters
    ----------
    L_in : :obj:`~equitorch.typing.DegreeRange`
        The input degree range.
    L_out : :obj:`~equitorch.typing.DegreeRange`
        The output degree range.
    in_channels : int
        The number of input channels.
    out_channels : int
        The number of output channels.
    external_weights : bool, optional
        Whether to use an external weights. Defaults to False.
        Default to False.
    channel_wise : bool, optional
        Whether to use an external weights. 
        Defaults to False.
    '''
    def __init__(self, 
                 L_in: DegreeRange, 
                 L_out: DegreeRange, 
                 in_channels: int,
                 out_channels: int,
                 external_weights: bool = False,
                 channel_wise: bool = False
                 ):
        assert in_channels == out_channels or not channel_wise
        super().__init__()
        self.L_in = check_degree_range(L_in)
        self.L_out = check_degree_range(L_out)
        self.in_channels = in_channels
        self.out_ms = num_orders_in(self.L_out)
        self.out_channels = out_channels 
        self.channel_wise = channel_wise

        Ms, ls, weight_sign, weight_index = _so2_indices(L_in, L_out)
        
        self.register_buffer('M_out', Ms[0])
        self.register_buffer('M_in', Ms[1])
        self.register_buffer('weight_sign', weight_sign)
        self.register_buffer('weight_index', weight_index)

        self.num_weights = weight_index.max().item()+1

        self.external_weights = external_weights
        
        if self.channel_wise:
            self.weight_shape = (self.num_weights, self.in_channels)
            if not self.external_weights:
                self.weight = nn.Parameter(
                    torch.randn(1, self.num_weights,
                                self.in_channels))
        else:
            self.weight_shape = (self.num_weights, self.in_channels, self.out_channels)
            if not self.external_weights:
                self.weight = nn.Parameter(
                    torch.randn(1, self.num_weights,
                                self.in_channels, 
                                self.out_channels)
                        * 2 / sqrt(in_channels + out_channels))

    def forward(self, x: Tensor, weight: Optional[Tensor]=None):
        r"""
        Applies the SO(2) linear operation to the input tensor.

        Parameters
        ----------
        x : :obj:`~torch.Tensor`
            The input tensor of shape :math:`(N, \text{num_orders_in}, C_{\text{in}})`, 
            where :math:`N` is the batch size, :math:`\text{num_orders_in}` 
            is the number of input orders, and :math:`C_{\text{in}}` is the number of channels.
            before passed into this function, :math:`x` must have been transformed by 
            :math:`\mathbf{D}_{\text{in}}`.
        weight : :obj:`~torch.Tensor`, optional
            The external weights to use for the linear operation. If `None`, the 
            internal weights will be used.
            The shape of the weights depends on the value of `channel_wise`. 
            If `channel_wise` is `True`, the shape should be :math:`(N, \text{num_weights}, C_{\text{in}})`.
            If `channel_wise` is `False`, the shape should be :math:`(N, \text{num_weights}, C_{\text{in}}, C_{\text{out}})`, 

        Returns
        -------
        :obj:`~torch.Tensor`
            The output tensor of shape :math:`(N, \text{num_orders_out}, C_{\text{out}})`.
            The returned feature should then be transformed by :math:`\mathbf{D}_{\text{out}}^\top`.

        Note
        ----
        If :obj:`external_weights` is :obj:`True`, the :obj:`weight` parameter must be provided. 
        If :obj:`external_weights` is :obj:`False`, the :obj:`weight` will still be used if provided.
        """
        if weight is None and not self.external_weights:
            weight = self.weight
        else:
            weight = weight.view(-1, *(self.weight_shape))
            
        X = x.index_select(dim=1,index=self.M_in) # N * Num * Ci 

        if self.channel_wise:
            W = weight.index_select(dim=1, index=self.weight_index) * self.weight_sign.view(1,-1,1) # N * Num * Ci (* Co)
            out = X * W
        else:
            W = weight.index_select(dim=1, index=self.weight_index) * self.weight_sign.view(1,-1,1,1) # N * Num * Ci (* Co)
            out = (X.unsqueeze(-2)@W).squeeze(-2)
        return scatter(out, index=self.M_out, dim=1, dim_size=self.out_ms)
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(L_in={self.L_in}, L_out={self.L_out}, in_channels={self.in_channels},\n  out_channels={self.out_channels}, channel_wise={self.channel_wise}, external_weights={self.external_weights}\n)'
