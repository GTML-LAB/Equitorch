"""Provides data transformation utilities, particularly for graph-based data compatible with PyTorch Geometric."""
"""
Some data transforms.
"""
from typing import Callable
from torch import Tensor
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform

from ..nn.functional.sphericals import spherical_harmonics



# Modified from https://pytorch-geometric.readthedocs.io/en/2.6.1/_modules/torch_geometric/transforms/radius_graph.html
@functional_transform('radius_graph_eqt')
class RadiusGraph(BaseTransform):
    r"""Creates edges based on node positions ``pos_attr`` to all points
    within a given cutoff distance (functional name: ``radius_graph_eqt``).

    Args:
        r (float): The cutoff distance.
        loop (bool, optional): If ``True``, the graph will contain self-loops.
            Defaults to ``False``.
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element. This flag is only needed for CUDA tensors.
            Defaults to ``32``.
        flow (str, optional): The flow direction when using in combination with
            message passing (``"source_to_target"`` or ``"target_to_source"``).
            Defaults to ``"source_to_target"``.
        pos_attr (str, optional): The attribute name for positions in the data.
            Defaults to ``"pos"``.
        edge_index_attr (str, optional): The attribute name for creating edge
            index in the data. Defaults to ``"edge_index"``.
        edge_vector_attr (str, optional): The attribute name for creating edge
            vectors in the data. Defaults to ``"edge_vec"``.
        num_workers (int, optional): Number of workers to use for computation.
            Has no effect in case batch is not ``None``, or the input lies on the GPU.
            Defaults to ``1``.

    Example:
        >>> N = 50
        >>> pos = torch.randn(N,3)
        >>> data = Data(pos=pos)
        >>> print(data)
        Data(pos=[50, 3])
        >>> data = RadiusGraph(0.5)(data)
        >>> print(data)
        Data(pos=[50, 3], edge_index=[2, 36], edge_vec=[36, 3])
    """
    def __init__(
        self,
        r: float,
        loop: bool = False,
        max_num_neighbors: int = 32,
        flow: str = 'source_to_target',
        pos_attr: str = 'pos',
        edge_index_attr: str = 'edge_index',
        edge_vector_attr: str = 'edge_vec',
        num_workers: int = 1,
    ) -> None:
        self.r = r
        self.loop = loop
        self.max_num_neighbors = max_num_neighbors
        self.flow = flow
        self.num_workers = num_workers

        self.pos_attr = pos_attr
        self.edge_index_attr = edge_index_attr
        self.edge_vector_attr = edge_vector_attr

    def forward(self, data: Data) -> Data:
        """Adds edge index and edge vectors to the data object.

        Args:
            data (torch_geometric.data.Data): The input data object.

        Returns:
            torch_geometric.data.Data: The data object with added ``edge_index``
                and ``edge_vec`` (or names specified by ``edge_index_attr`` and
                ``edge_vector_attr`` respectively).
        """
        # assert data.pos is not None


        pos = data.__getattr__(self.pos_attr)
        edge_index = torch_geometric.nn.radius_graph(
            pos,
            self.r,
            data.batch,
            self.loop,
            max_num_neighbors=self.max_num_neighbors,
            flow=self.flow,
            num_workers=self.num_workers,
        )
        data.__setattr__(self.edge_index_attr, edge_index)

        data.__setattr__(self.edge_vector_attr, 
            data.pos[edge_index[1]] - data.pos[edge_index[0]]
        )

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(r={self.r})'
    

@functional_transform('add_vector_norm')
class AddVectorNorm(BaseTransform):
    r"""Computes the norm of a vector attribute and adds it to the data object.

    This transform is useful for obtaining scalar distance information from
    vector representations, commonly used in graph neural networks for edge
    features.

    (functional name: ``add_vector_norm``)

    Args:
        vector_attr (str, optional): The attribute name for the input vector
            in the :class:`torch_geometric.data.Data` object. Defaults to ``'edge_vec'``.
        norm_attr (str, optional): The attribute name under which the computed
            norm will be stored in the :class:`torch_geometric.data.Data` object.
            Defaults to ``'edge_norm'``.
    """
    def __init__(
        self,
        vector_attr: str = 'edge_vec',
        norm_attr: str = 'edge_norm',
    ) -> None:

        self.vector_attr = vector_attr
        self.norm_attr = norm_attr

    def forward(self, data: Data) -> Data:
        """Adds the norm of the specified vector attribute to the data object.

        Args:
            data (torch_geometric.data.Data): The input data object, which must
                contain the attribute specified by ``self.vector_attr``.

        Returns:
            torch_geometric.data.Data: The input data object, augmented with a new
                attribute (specified by ``self.norm_attr``) containing the L2 norm
                of the ``self.vector_attr``.
        """
        vec = data.__getattr__(self.vector_attr)
        data.__setattr__(self.norm_attr, vec.norm(dim=-1))

        return data

    
@functional_transform('add_spherical_harmonics')
class AddSphericalHarmonics(BaseTransform):
    r"""Creates spherical harmonics embedding based on direction vectors ``vector_attr``
    (functional name: ``add_spherical_harmonics``).

    Args:
        l_max (int): The maximum degree of spherical harmonics.
        vector_attr (str, optional): The attribute name for direction vectors
            in the data. Defaults to ``"edge_vec"``.
        sh_attr (str, optional): The attribute name for storing spherical
            harmonics in the data. Defaults to ``"edge_sh"``.
        integral_normalize (bool, optional): Whether to normalize the spherical
            harmonics by :math:`\sqrt{4\pi / (2l+1)}`. Defaults to ``False``.

    Example:
        >>> print(data)
        Data(pos=[50, 3], edge_index=[2, 36], edge_vec=[36, 3])
        >>> data = AddSphericalHarmonics(l_max=3)(data)
        >>> print(data)
        Data(pos=[50, 3], edge_index=[2, 36], edge_vec=[36, 3], edge_sh=[36, 16])
    """
    def __init__(
        self,
        l_max: int,
        vector_attr: str = 'edge_vec',
        sh_attr: str = 'edge_sh',
        integral_normalize: bool = False
    ) -> None:

        self.l_max = l_max
        self.vector_attr = vector_attr
        self.sh_attr = sh_attr
        self.integral_normalize = integral_normalize

    def forward(self, data: Data) -> Data:
        """Adds spherical harmonics of the specified vector attribute to the data object.

        Args:
            data (torch_geometric.data.Data): The input data object, which must
                contain the attribute specified by ``self.vector_attr``.

        Returns:
            torch_geometric.data.Data: The data object with added spherical harmonics
                stored in the attribute specified by ``self.sh_attr``.
        """
        vec = data.__getattr__(self.vector_attr)
        data.__setattr__(self.sh_attr, spherical_harmonics(vec, self.l_max, integral_normalize=self.integral_normalize))

        return data

@functional_transform('add_vector_length_embedding')   
class AddVectorLengthEmbedding(BaseTransform):
    r"""Add the length embedding for vectors.

    (functional name: ``add_vector_length_embedding``)

    Args:
        emb (Callable[[torch.Tensor,], torch.Tensor]): The length embedding operation.
            This should be a callable that takes a tensor of vector norms
            and returns their embeddings.
        vector_attr (str, optional): The attribute name for direction vectors
            in the data. Defaults to ``"edge_vec"``.
        embedding_attr (str, optional): The attribute name for storing the length
            embedding. Defaults to ``"edge_emb"``.

    Example:
        >>> print(data)
        Data(pos=[50, 3], edge_index=[2, 36], edge_vec=[36, 3])
        >>> data = AddVectorLengthEmbedding(BesselBasis(r_max=5.0, num_basis=8))(data)
        >>> print(data)
        Data(pos=[50, 3], edge_index=[2, 36], edge_vec=[36, 3], edge_emb=[36, 8])
    """
    def __init__(
        self,
        emb: Callable[[Tensor], Tensor],
        vector_attr: str = 'edge_vec',
        embedding_attr: str = 'edge_emb'
    ) -> None:

        self.emb = emb
        self.vector_attr = vector_attr
        self.embedding_attr = embedding_attr

    def forward(self, data: Data) -> Data:
        """Adds the length embedding of the specified vector attribute to the data object.

        Args:
            data (torch_geometric.data.Data): The input data object, which must
                contain the attribute specified by ``self.vector_attr``.

        Returns:
            torch_geometric.data.Data: The data object with the added length embedding
                stored in the attribute specified by ``self.embedding_attr``.
        """
        vec = data.__getattr__(self.vector_attr)
        
        data.__setattr__(self.embedding_attr, self.emb(vec.norm(dim=-1)))

        return data
   
__all__ = [
    'RadiusGraph',
    'AddSphericalHarmonics',
    'AddVectorNorm',
]