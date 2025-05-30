import torch

def initialize_tensor_product(weight, feature_mode, gain=1, channel_normed=False):
    r"""Initialize weights for tensor product operations.
    
    This function initializes weights for tensor product operations with different
    feature modes. The initialization uses a uniform distribution with bounds
    calculated based on the feature mode and whether channel normalization is used.
    
    For 'uvw' mode:
    
    .. math::
        a = \begin{cases}
        \sqrt{3} \cdot \text{gain}, & \text{if channel_normed} = \text{True} \\
        \sqrt{\frac{3}{\text{fan_in}}} \cdot \text{gain}, & \text{otherwise}
        \end{cases}
    
    where :math:`\text{fan_in} = \text{weight.shape[-2]} \cdot \text{weight.shape[-3]}`
    
    For 'uuu' mode:
    
    .. math::
        a = \sqrt{3} \cdot \text{gain}
    
    Args:
        weight (torch.Tensor): The weight tensor to initialize.
        feature_mode (str): The feature mode for initialization. Must be one of ['uvw', 'uuu'].
        gain (float, optional): The gain factor to apply. Default is 1.
        channel_normed (bool, optional): Whether channel normalization is used. Default is False.
    
    Raises:
        ValueError: If an unknown feature_mode is provided.
    """

    if feature_mode == 'uvw':
        fan_out = weight.shape[-1]
        fan_in = weight.shape[-2] * weight.shape[-3]
        # Skip fan_in normalization if channel_normed is True
        a = (3 ** 0.5 * gain) if channel_normed else (3.0 / fan_in) ** 0.5 * gain
    elif feature_mode == 'uuu':
        # 'uuu' mode doesn't have channel normalization based on fan_in
        a = 3 ** 0.5 * gain
    else:
        raise ValueError(f"Unknown feature_mode '{feature_mode}' for initialize_tensor_product")

    torch.nn.init.uniform_(weight, -a, a)

def initialize_so3_so2_linear(weight, feature_mode, gain=1, channel_normed=False):
    r"""Initialize weights for SO(3) or SO(2) linear operations.
    
    This function initializes weights for SO(3) or SO(2) linear operations with different
    feature modes. The initialization uses a uniform distribution with bounds
    calculated based on the feature mode and whether channel normalization is used.
    
    For 'uv' mode:
    
    .. math::
        a = \begin{cases}
        \sqrt{3} \cdot \text{gain}, & \text{if channel_normed} = \text{True} \\
        \sqrt{\frac{3}{\text{fan_in}}} \cdot \text{gain}, & \text{otherwise}
        \end{cases}
    
    where :math:`\text{fan_in} = \text{weight.shape[-2]}`
    
    For 'uu' mode:
    
    .. math::
        a = \sqrt{3} \cdot \text{gain}
    
    Args:
        weight (torch.Tensor): The weight tensor to initialize.
        feature_mode (str): The feature mode for initialization. Must be one of ['uv', 'uu'].
        gain (float, optional): The gain factor to apply. Default is 1.
        channel_normed (bool, optional): Whether channel normalization is used. Default is False.
    
    Raises:
        ValueError: If an unknown feature_mode is provided.
    """

    if feature_mode == 'uv':
        fan_out = weight.shape[-1]
        fan_in = weight.shape[-2]
        # Skip fan_in normalization if channel_normed is True
        a = (3 ** 0.5 * gain) if channel_normed else (3.0 / fan_in) ** 0.5 * gain
    elif feature_mode == 'uu':
        # 'uu' mode doesn't have channel normalization based on fan_in
        a = 3 ** 0.5 * gain
    else:
        raise ValueError(f"Unknown feature_mode '{feature_mode}' for initialize_so3_so2_linear")

    torch.nn.init.uniform_(weight, -a, a)

def initialize_linear(weight, gain=1, channel_normed=False):
    r"""Initialize weights for standard linear operations.
    
    This function initializes weights for standard linear operations using a uniform
    distribution with bounds calculated based on whether channel normalization is used.
    
    .. math::
        a = \begin{cases}
        \sqrt{3} \cdot \text{gain}, & \text{if channel_normed} = \text{True} \\
        \sqrt{\frac{3}{\text{fan_in}}} \cdot \text{gain}, & \text{otherwise}
        \end{cases}
    
    where :math:`\text{fan_in} = \text{weight.shape[-2]}`
    
    Args:
        weight (torch.Tensor): The weight tensor to initialize.
        gain (float, optional): The gain factor to apply. Default is 1.
        channel_normed (bool, optional): Whether channel normalization is used. Default is False.
    """

    fan_out = weight.shape[-1]
    fan_in = weight.shape[-2]
    if not channel_normed:
        a = (3.0 / (fan_in)) ** 0.5 * gain
    else:
        a = 3.0 ** 0.5 * gain
    torch.nn.init.uniform_(weight, -a, a)
