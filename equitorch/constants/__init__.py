"""Defines mathematical constants and utility functions."""
from typing import Union

import numpy as np
import sympy
from sympy import MutableDenseNDimArray, Matrix
from sympy.physics.wigner import clebsch_gordan
from sympy import S

import functools

@functools.lru_cache(maxsize=None)
def j_matrix(l: int):
    r"""Computes the Wigner D-matrix for the rotation that exchanges y-z axes and reverses x-axis.
    
    This function calculates the Wigner D-matrix corresponding to a specific rotation
    transformation that exchanges the y and z axes while reversing the direction of the x-axis.
    It is used in general Wigner D-matrix calculations.
    
    Args:
        l: The angular momentum quantum number.
        
    Returns:
        sympy.Matrix: The Wigner D-matrix for the specified rotation transformation.
    """
    if l == 0:
        return Matrix([[1]])
    if l == 1:
        return Matrix(
            [[ 0,  1,  0],
             [ 1,  0,  0],
             [ 0,  0, -1]])
    else:
        cg = so3_clebsch_gordan(l,1,l-1)
        j = j_matrix(l-1)
        return np.einsum('iI,jJ,kij,KIJ->kK',j_matrix(1), j, cg, cg)
    

# adapted and modified from e3nn
@functools.lru_cache(maxsize=None)
def so3_clebsch_gordan(l, l1, l2):
    r"""Computes the Clebsch-Gordan coefficients for SO(3) group.
    
    This function calculates the Clebsch-Gordan coefficients for the SO(3) group,
    which are used to decompose the tensor product of two irreducible representations
    into a direct sum of irreducible representations.
    
    Args:
        l: An integer representing the angular momentum quantum number of the resulting representation.
        l1: An integer representing the angular momentum quantum number of the first representation.
        l2: An integer representing the angular momentum quantum number of the second representation.
        
    Returns:
        sympy.Array: The Clebsch-Gordan coefficients matrix for the specified angular momenta.
    """
    Q1 = _change_basis_real_to_complex(l1)
    Q2 = _change_basis_real_to_complex(l2)
    QT = sympy.conjugate(_change_basis_real_to_complex(l)).transpose()
    C = _su2_clebsch_gordan(l, l1, l2)
    C = np.einsum("mn,nik,ij,kl->mjl", QT, C, Q1, Q2)
    return sympy.sympify(C)

@functools.lru_cache(maxsize=None)
def _change_basis_real_to_complex(l: int):
    sqrt2 = sympy.sqrt(2)
    q = MutableDenseNDimArray.zeros(2 * l + 1, 2 * l + 1)
    for m in range(-l, 0):
        q[l + m, l + abs(m)] = sqrt2 / 2
        q[l + m, l - abs(m)] = -sqrt2 * sympy.I / 2
    q[l, l] = 1
    for m in range(1, l + 1):
        q[l + m, l + abs(m)] = (-1) ** S(m) * sqrt2 / 2
        q[l + m, l - abs(m)] = sympy.I * (-1) ** S(m) * sqrt2 / 2
    q = (-sympy.I) ** S(l) * q  # Added factor of sympy.I**l to make the Clebsch-Gordan coefficients real
    return q

@functools.lru_cache(maxsize=None)
def _su2_clebsch_gordan(j3: Union[int, float], j1: Union[int, float], j2: Union[int, float]):
    r"""Calculates the Clebsch-Gordon matrix
    for SU(2) coupling j1 and j2 to give j3.
    Parameters
    ----------
    j3 : float
        Total angular momentum 3.
    j1 : float
        Total angular momentum 1.
    j2 : float
        Total angular momentum 2.
    Returns
    -------
    cg_matrix : MutableDenseNDimArray
        Requested Clebsch-Gordan matrix.
    """
    assert isinstance(j1, (int, float))
    assert isinstance(j2, (int, float))
    assert isinstance(j3, (int, float))
    mat = MutableDenseNDimArray.zeros(int(2 * j3 + 1), int(2 * j1 + 1), int(2 * j2 + 1))
    if int(2 * j3) in range(int(2 * abs(j1 - j2)), int(2 * (j1 + j2)) + 1, 2):
        for twice_m1 in (x for x in range(-int(2 * j1), int(2 * j1) + 1, 2)):
            for twice_m2 in (x for x in range(-int(2 * j2), int(2 * j2) + 1, 2)):
                if abs(twice_m1 + twice_m2) <= 2*j3:
                    mat[j3 + int(twice_m1/2 + twice_m2/2), int(j1 + twice_m1/2), int(j2 + twice_m2/2)] = clebsch_gordan(
                        j1, j2, j3, S(twice_m1)/2, S(twice_m2)/2, S(twice_m1+twice_m2)/2
                    )
    return mat

# The function _su2_clebsch_gordan is modified from
# QuTiP: Quantum Toolbox in Python.
# Key modifications are: (1) set the j3 to the first axis
#                        (2) use MutableDenseNDimArray instead of numpy array
#                        (3) use sympy.physics.wigner.clebsch_gordan
#                            instead of qutip.utilities._su2_clebsch_gordan_coeff
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
