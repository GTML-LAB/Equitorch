"""Provides functional implementations of equivariant neural network operations."""
from .linears import (
	# TensorProductU1UDummy,
	tensor_product_u1u,
	so3_linear_uu,
	# TensorProduct1UUDummy,
	tensor_product_1uu,
	# TensorProductUU1Dummy,
	tensor_product_uu1,
	# TensorProductU1VDummy,
	tensor_product_u1v,
	so3_linear_uv,
	# TensorProduct1VUDummy,
	tensor_product_1vu,
	# TensorProductVU1Dummy,
	tensor_product_vu1,
	# IrrepWiseLinear,
	irrep_wise_linear,
	# IrrepsLinear,
	irreps_linear,
	so2_linear_uu,
	so2_linear_uv
)
from .sphericals import (
	spherical_harmonics,
	xyz_to_spherical,
	spherical_to_xyz,
	xyz_to_sincos
)
from .normalization import (
	batch_rms_norm,
	layer_rms_norm
)
from .sparse_scale import (
	# SparseScale,
	sparse_scale
)
from .tensor_products import (
	# TensorProductUUUDummy,
	tensor_product_uuu,
	# TensorProductUVWDummy,
	tensor_product_uvw,
	# TensorDotUU,
	tensor_dot_uu,
	# TensorDotUV,
	tensor_dot_uv
)
from .sparse_product import (
	# SparseMul,
	sparse_mul,
	# SparseOuter,
	sparse_outer,
	# SparseInner,
	sparse_inner,
	# SparseVecMat,
	sparse_vecmat,
	# SparseVecSca,
	sparse_vecsca,
	# SparseScaVec,
	sparse_scavec,
	# SparseMatTVec,
	sparse_mat_t_vec
)
from .wigner_d import (
	sparse_wigner_rotation,
	dense_wigner_rotation,
	wigner_d_matrix,
	align_to_z_wigner_d
)
from .cutoffs import (
	radial_standarize,
	polynomial_cutoff,
	cosine_cutoff,
	mollifier_cutoff
)
from .activations import (
	gating
)
from .angular import (
	sincos
)
from .norm import (
	# SquaredNorm,
	squared_norm,
	# Norm,
	norm,
	# ChannelMeanSquaredNorm,
	channel_mean_squared_norm,
	# BatchMeanSquaredNorm,
	batch_mean_squared_norm
)
from .dropout import (
	irrep_wise_dropout
)
from .rotations import (
	angles_to_matrix
)

__all__ = [
	# "TensorProductU1UDummy",
	"tensor_product_u1u",
	"so3_linear_uu",
	# "TensorProduct1UUDummy",
	"tensor_product_1uu",
	# "TensorProductUU1Dummy",
	"tensor_product_uu1",
	# "TensorProductU1VDummy",
	"tensor_product_u1v",
	"so3_linear_uv",
	# "TensorProduct1VUDummy",
	"tensor_product_1vu",
	# "TensorProductVU1Dummy",
	"tensor_product_vu1",
	# "IrrepWiseLinear",
	"irrep_wise_linear",
	# "IrrepsLinear",
	"irreps_linear",
	"so2_linear_uu",
	"so2_linear_uv",
	"spherical_harmonics",
	"xyz_to_spherical",
	"spherical_to_xyz",
	"xyz_to_sincos",
	"batch_rms_norm",
	"layer_rms_norm",
	# "SparseScale",
	"sparse_scale",
	# "TensorProductUUUDummy",
	"tensor_product_uuu",
	# "TensorProductUVWDummy",
	"tensor_product_uvw",
	# "TensorDotUU",
	"tensor_dot_uu",
	# "TensorDotUV",
	"tensor_dot_uv",
	# "SparseMul",
	"sparse_mul",
	# "SparseOuter",
	"sparse_outer",
	# "SparseInner",
	"sparse_inner",
	# "SparseVecMat",
	"sparse_vecmat",
	# "SparseVecSca",
	"sparse_vecsca",
	# "SparseScaVec",
	"sparse_scavec",
	# "SparseMatTVec",
	"sparse_mat_t_vec",
	"sparse_wigner_rotation",
	"dense_wigner_rotation",
	"wigner_d_matrix",
	"align_to_z_wigner_d",
	"radial_standarize",
	"polynomial_cutoff",
	"cosine_cutoff",
	"mollifier_cutoff",
	"gating",
	"sincos",
	# "SquaredNorm",
	"squared_norm",
	# "Norm",
	"norm",
	# "ChannelMeanSquaredNorm",
	"channel_mean_squared_norm",
	# "BatchMeanSquaredNorm",
	"batch_mean_squared_norm",
	"irrep_wise_dropout",
	"angles_to_matrix"
]
