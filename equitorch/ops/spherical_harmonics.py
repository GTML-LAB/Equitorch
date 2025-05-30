import math

import torch

# def _generate_spherical_harmonics(l_max, device=None):  # pragma: no cover
#     r"""code used to generate the code of _spherical_harmonics

#     based on `wigner_3j`
#     """
#     torch.set_default_dtype(torch.float64)

#     print("""x = X.narrow(dim,0,1)\ny = X.narrow(dim,1,1)\nz = X.narrow(dim,2,1)\n""")

#     def to_frac(x: float):
#         from fractions import Fraction

#         s = 1 if x >= 0 else -1
#         x = x**2
#         x = Fraction(x).limit_denominator()
#         x = s * sympy.sqrt(x)
#         x = sympy.simplify(x)
#         return x

#     print("sh_0_0 = torch.ones_like(x)")
#     print("if l_max == 0:")
#     print("    return torch.cat([")
#     print("        sh_0_0,")
#     print("    ], dim=dim)")
#     print()

#     x_var, y_var, z_var = sympy.symbols("y z x")
#     polynomials = [sympy.sqrt(3) * x_var, sympy.sqrt(3) * y_var, sympy.sqrt(3) * z_var]
    
#     def sub_z1(p, names, polynormz):
#         p = p.subs(x_var, 0).subs(y_var, 1).subs(z_var, 0)
#         for n, c in zip(names, polynormz):
#             p = p.subs(n, c)
#         return p

#     poly_evalz = [sub_z1(p, [], []) for p in polynomials]

#     for l in range(1, l_max + 1):
#         sh_variables = sympy.symbols(" ".join(f"sh_{l}_{m}" for m in range(2 * l + 1)))

#         for n, p in zip(sh_variables, polynomials):
#             print(f"{n} = {pycode(p)}")

#         print(f"if l_max == {l}:")
#         u = ",\n        ".join(", ".join(f"sh_{j}_{m}" for m in range(2 * j + 1)) for j in range(l + 1))
#         print(f"    return torch.cat([\n        {u}\n    ], dim=dim)")
#         print()

#         if l == l_max:
#             break

#         polynomials = [
#             sum(to_frac(c.item()) * v * sh for cj, v in zip(cij, [x_var, y_var, z_var]) for c, sh in zip(cj, sh_variables))
#             for cij in o3.wigner_3j(l + 1, 1, l, device=device)
#         ]

#         poly_evalz = [sub_z1(p, sh_variables, poly_evalz) for p in polynomials]
#         norm = sympy.sqrt(sum(p**2 for p in poly_evalz))
#         polynomials = [sympy.sqrt(2 * l + 3) * p / norm for p in polynomials]
#         poly_evalz = [sympy.sqrt(2 * l + 3) * p / norm for p in poly_evalz]

#         polynomials = [sympy.simplify(p, full=True) for p in polynomials]

@torch.jit.script
def spherical_harmonics(input: torch.Tensor, l_max:int, dim:int=-1):

    x = input.narrow(dim,0,1)
    y = input.narrow(dim,1,1)
    z = input.narrow(dim,2,1)

    sh_0_0 = torch.ones_like(x)
    if l_max == 0:
        return torch.cat([
            sh_0_0,
        ], dim=dim)

    sh_1_0 = math.sqrt(3)*y
    sh_1_1 = math.sqrt(3)*z
    sh_1_2 = math.sqrt(3)*x
    if l_max == 1:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2
        ], dim=dim)

    sh_2_0 = (1/2)*math.sqrt(5)*(sh_1_0*x + sh_1_2*y)
    sh_2_1 = (1/2)*math.sqrt(5)*(sh_1_0*z + sh_1_1*y)
    sh_2_2 = (1/6)*math.sqrt(15)*(-sh_1_0*y + 2*sh_1_1*z - sh_1_2*x)
    sh_2_3 = (1/2)*math.sqrt(5)*(sh_1_1*x + sh_1_2*z)
    sh_2_4 = (1/2)*math.sqrt(5)*(-sh_1_0*y + sh_1_2*x)
    if l_max == 2:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4
        ], dim=dim)

    sh_3_0 = (1/6)*math.sqrt(42)*(sh_2_0*x + sh_2_4*y)
    sh_3_1 = (1/3)*math.sqrt(7)*(sh_2_0*z + sh_2_1*x + sh_2_3*y)
    sh_3_2 = -1/30*math.sqrt(70)*sh_2_0*x + (2/15)*math.sqrt(70)*sh_2_1*z + (1/15)*math.sqrt(210)*sh_2_2*y + (1/30)*math.sqrt(70)*sh_2_4*y
    sh_3_3 = -1/15*math.sqrt(105)*sh_2_1*y + (1/5)*math.sqrt(35)*sh_2_2*z - 1/15*math.sqrt(105)*sh_2_3*x
    sh_3_4 = -1/30*math.sqrt(70)*sh_2_0*y + (1/15)*math.sqrt(210)*sh_2_2*x + (2/15)*math.sqrt(70)*sh_2_3*z - 1/30*math.sqrt(70)*sh_2_4*x
    sh_3_5 = (1/3)*math.sqrt(7)*(-sh_2_1*y + sh_2_3*x + sh_2_4*z)
    sh_3_6 = (1/6)*math.sqrt(42)*(-sh_2_0*y + sh_2_4*x)
    if l_max == 3:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6
        ], dim=dim)

    sh_4_0 = (3/4)*math.sqrt(2)*(sh_3_0*x + sh_3_6*y)
    sh_4_1 = (3/4)*sh_3_0*z + (3/8)*math.sqrt(6)*sh_3_1*x + (3/8)*math.sqrt(6)*sh_3_5*y
    sh_4_2 = -3/56*math.sqrt(14)*sh_3_0*x + (3/14)*math.sqrt(21)*sh_3_1*z + (3/56)*math.sqrt(210)*sh_3_2*x + (3/56)*math.sqrt(210)*sh_3_4*y + (3/56)*math.sqrt(14)*sh_3_6*y
    sh_4_3 = -3/56*math.sqrt(42)*sh_3_1*x + (3/28)*math.sqrt(105)*sh_3_2*z + (3/28)*math.sqrt(70)*sh_3_3*y + (3/56)*math.sqrt(42)*sh_3_5*y
    sh_4_4 = -3/28*math.sqrt(42)*sh_3_2*y + (3/7)*math.sqrt(7)*sh_3_3*z - 3/28*math.sqrt(42)*sh_3_4*x
    sh_4_5 = -3/56*math.sqrt(42)*sh_3_1*y + (3/28)*math.sqrt(70)*sh_3_3*x + (3/28)*math.sqrt(105)*sh_3_4*z - 3/56*math.sqrt(42)*sh_3_5*x
    sh_4_6 = -3/56*math.sqrt(14)*sh_3_0*y - 3/56*math.sqrt(210)*sh_3_2*y + (3/56)*math.sqrt(210)*sh_3_4*x + (3/14)*math.sqrt(21)*sh_3_5*z - 3/56*math.sqrt(14)*sh_3_6*x
    sh_4_7 = -3/8*math.sqrt(6)*sh_3_1*y + (3/8)*math.sqrt(6)*sh_3_5*x + (3/4)*sh_3_6*z
    sh_4_8 = (3/4)*math.sqrt(2)*(-sh_3_0*y + sh_3_6*x)
    if l_max == 4:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8
        ], dim=dim)

    sh_5_0 = (1/10)*math.sqrt(110)*(sh_4_0*x + sh_4_8*y)
    sh_5_1 = (1/5)*math.sqrt(11)*sh_4_0*z + (1/5)*math.sqrt(22)*sh_4_1*x + (1/5)*math.sqrt(22)*sh_4_7*y
    sh_5_2 = -1/30*math.sqrt(22)*sh_4_0*x + (4/15)*math.sqrt(11)*sh_4_1*z + (1/15)*math.sqrt(154)*sh_4_2*x + (1/15)*math.sqrt(154)*sh_4_6*y + (1/30)*math.sqrt(22)*sh_4_8*y
    sh_5_3 = -1/30*math.sqrt(66)*sh_4_1*x + (1/15)*math.sqrt(231)*sh_4_2*z + (1/30)*math.sqrt(462)*sh_4_3*x + (1/30)*math.sqrt(462)*sh_4_5*y + (1/30)*math.sqrt(66)*sh_4_7*y
    sh_5_4 = -1/15*math.sqrt(33)*sh_4_2*x + (2/15)*math.sqrt(66)*sh_4_3*z + (1/15)*math.sqrt(165)*sh_4_4*y + (1/15)*math.sqrt(33)*sh_4_6*y
    sh_5_5 = -1/15*math.sqrt(110)*sh_4_3*y + (1/3)*math.sqrt(11)*sh_4_4*z - 1/15*math.sqrt(110)*sh_4_5*x
    sh_5_6 = -1/15*math.sqrt(33)*sh_4_2*y + (1/15)*math.sqrt(165)*sh_4_4*x + (2/15)*math.sqrt(66)*sh_4_5*z - 1/15*math.sqrt(33)*sh_4_6*x
    sh_5_7 = -1/30*math.sqrt(66)*sh_4_1*y - 1/30*math.sqrt(462)*sh_4_3*y + (1/30)*math.sqrt(462)*sh_4_5*x + (1/15)*math.sqrt(231)*sh_4_6*z - 1/30*math.sqrt(66)*sh_4_7*x
    sh_5_8 = -1/30*math.sqrt(22)*sh_4_0*y - 1/15*math.sqrt(154)*sh_4_2*y + (1/15)*math.sqrt(154)*sh_4_6*x + (4/15)*math.sqrt(11)*sh_4_7*z - 1/30*math.sqrt(22)*sh_4_8*x
    sh_5_9 = -1/5*math.sqrt(22)*sh_4_1*y + (1/5)*math.sqrt(22)*sh_4_7*x + (1/5)*math.sqrt(11)*sh_4_8*z
    sh_5_10 = (1/10)*math.sqrt(110)*(-sh_4_0*y + sh_4_8*x)
    if l_max == 5:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10
        ], dim=dim)

    sh_6_0 = (1/6)*math.sqrt(39)*(sh_5_0*x + sh_5_10*y)
    sh_6_1 = (1/6)*math.sqrt(13)*sh_5_0*z + (1/12)*math.sqrt(130)*sh_5_1*x + (1/12)*math.sqrt(130)*sh_5_9*y
    sh_6_2 = -1/132*math.sqrt(286)*sh_5_0*x + (1/33)*math.sqrt(715)*sh_5_1*z + (1/132)*math.sqrt(286)*sh_5_10*y + (1/44)*math.sqrt(1430)*sh_5_2*x + (1/44)*math.sqrt(1430)*sh_5_8*y
    sh_6_3 = -1/132*math.sqrt(858)*sh_5_1*x + (1/22)*math.sqrt(429)*sh_5_2*z + (1/22)*math.sqrt(286)*sh_5_3*x + (1/22)*math.sqrt(286)*sh_5_7*y + (1/132)*math.sqrt(858)*sh_5_9*y
    sh_6_4 = -1/66*math.sqrt(429)*sh_5_2*x + (2/33)*math.sqrt(286)*sh_5_3*z + (1/66)*math.sqrt(2002)*sh_5_4*x + (1/66)*math.sqrt(2002)*sh_5_6*y + (1/66)*math.sqrt(429)*sh_5_8*y
    sh_6_5 = -1/66*math.sqrt(715)*sh_5_3*x + (1/66)*math.sqrt(5005)*sh_5_4*z + (1/66)*math.sqrt(3003)*sh_5_5*y + (1/66)*math.sqrt(715)*sh_5_7*y
    sh_6_6 = -1/66*math.sqrt(2145)*sh_5_4*y + (1/11)*math.sqrt(143)*sh_5_5*z - 1/66*math.sqrt(2145)*sh_5_6*x
    sh_6_7 = -1/66*math.sqrt(715)*sh_5_3*y + (1/66)*math.sqrt(3003)*sh_5_5*x + (1/66)*math.sqrt(5005)*sh_5_6*z - 1/66*math.sqrt(715)*sh_5_7*x
    sh_6_8 = -1/66*math.sqrt(429)*sh_5_2*y - 1/66*math.sqrt(2002)*sh_5_4*y + (1/66)*math.sqrt(2002)*sh_5_6*x + (2/33)*math.sqrt(286)*sh_5_7*z - 1/66*math.sqrt(429)*sh_5_8*x
    sh_6_9 = -1/132*math.sqrt(858)*sh_5_1*y - 1/22*math.sqrt(286)*sh_5_3*y + (1/22)*math.sqrt(286)*sh_5_7*x + (1/22)*math.sqrt(429)*sh_5_8*z - 1/132*math.sqrt(858)*sh_5_9*x
    sh_6_10 = -1/132*math.sqrt(286)*sh_5_0*y - 1/132*math.sqrt(286)*sh_5_10*x - 1/44*math.sqrt(1430)*sh_5_2*y + (1/44)*math.sqrt(1430)*sh_5_8*x + (1/33)*math.sqrt(715)*sh_5_9*z
    sh_6_11 = -1/12*math.sqrt(130)*sh_5_1*y + (1/6)*math.sqrt(13)*sh_5_10*z + (1/12)*math.sqrt(130)*sh_5_9*x
    sh_6_12 = (1/6)*math.sqrt(39)*(-sh_5_0*y + sh_5_10*x)
    if l_max == 6:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12
        ], dim=dim)

    sh_7_0 = (1/14)*math.sqrt(210)*(sh_6_0*x + sh_6_12*y)
    sh_7_1 = (1/7)*math.sqrt(15)*sh_6_0*z + (3/7)*math.sqrt(5)*sh_6_1*x + (3/7)*math.sqrt(5)*sh_6_11*y
    sh_7_2 = -1/182*math.sqrt(390)*sh_6_0*x + (6/91)*math.sqrt(130)*sh_6_1*z + (3/91)*math.sqrt(715)*sh_6_10*y + (1/182)*math.sqrt(390)*sh_6_12*y + (3/91)*math.sqrt(715)*sh_6_2*x
    sh_7_3 = -3/182*math.sqrt(130)*sh_6_1*x + (3/182)*math.sqrt(130)*sh_6_11*y + (3/91)*math.sqrt(715)*sh_6_2*z + (5/182)*math.sqrt(858)*sh_6_3*x + (5/182)*math.sqrt(858)*sh_6_9*y
    sh_7_4 = (3/91)*math.sqrt(65)*sh_6_10*y - 3/91*math.sqrt(65)*sh_6_2*x + (10/91)*math.sqrt(78)*sh_6_3*z + (15/182)*math.sqrt(78)*sh_6_4*x + (15/182)*math.sqrt(78)*sh_6_8*y
    sh_7_5 = -5/91*math.sqrt(39)*sh_6_3*x + (15/91)*math.sqrt(39)*sh_6_4*z + (3/91)*math.sqrt(390)*sh_6_5*x + (3/91)*math.sqrt(390)*sh_6_7*y + (5/91)*math.sqrt(39)*sh_6_9*y
    sh_7_6 = -15/182*math.sqrt(26)*sh_6_4*x + (12/91)*math.sqrt(65)*sh_6_5*z + (2/91)*math.sqrt(1365)*sh_6_6*y + (15/182)*math.sqrt(26)*sh_6_8*y
    sh_7_7 = -3/91*math.sqrt(455)*sh_6_5*y + (1/13)*math.sqrt(195)*sh_6_6*z - 3/91*math.sqrt(455)*sh_6_7*x
    sh_7_8 = -15/182*math.sqrt(26)*sh_6_4*y + (2/91)*math.sqrt(1365)*sh_6_6*x + (12/91)*math.sqrt(65)*sh_6_7*z - 15/182*math.sqrt(26)*sh_6_8*x
    sh_7_9 = -5/91*math.sqrt(39)*sh_6_3*y - 3/91*math.sqrt(390)*sh_6_5*y + (3/91)*math.sqrt(390)*sh_6_7*x + (15/91)*math.sqrt(39)*sh_6_8*z - 5/91*math.sqrt(39)*sh_6_9*x
    sh_7_10 = -3/91*math.sqrt(65)*sh_6_10*x - 3/91*math.sqrt(65)*sh_6_2*y - 15/182*math.sqrt(78)*sh_6_4*y + (15/182)*math.sqrt(78)*sh_6_8*x + (10/91)*math.sqrt(78)*sh_6_9*z
    sh_7_11 = -3/182*math.sqrt(130)*sh_6_1*y + (3/91)*math.sqrt(715)*sh_6_10*z - 3/182*math.sqrt(130)*sh_6_11*x - 5/182*math.sqrt(858)*sh_6_3*y + (5/182)*math.sqrt(858)*sh_6_9*x
    sh_7_12 = -1/182*math.sqrt(390)*sh_6_0*y + (3/91)*math.sqrt(715)*sh_6_10*x + (6/91)*math.sqrt(130)*sh_6_11*z - 1/182*math.sqrt(390)*sh_6_12*x - 3/91*math.sqrt(715)*sh_6_2*y
    sh_7_13 = -3/7*math.sqrt(5)*sh_6_1*y + (3/7)*math.sqrt(5)*sh_6_11*x + (1/7)*math.sqrt(15)*sh_6_12*z
    sh_7_14 = (1/14)*math.sqrt(210)*(-sh_6_0*y + sh_6_12*x)
    if l_max == 7:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14
        ], dim=dim)

    sh_8_0 = (1/4)*math.sqrt(17)*(sh_7_0*x + sh_7_14*y)
    sh_8_1 = (1/8)*math.sqrt(17)*sh_7_0*z + (1/16)*math.sqrt(238)*sh_7_1*x + (1/16)*math.sqrt(238)*sh_7_13*y
    sh_8_2 = -1/240*math.sqrt(510)*sh_7_0*x + (1/60)*math.sqrt(1785)*sh_7_1*z + (1/240)*math.sqrt(46410)*sh_7_12*y + (1/240)*math.sqrt(510)*sh_7_14*y + (1/240)*math.sqrt(46410)*sh_7_2*x
    sh_8_3 = (1/80)*math.sqrt(2)*(-math.sqrt(85)*sh_7_1*x + math.sqrt(2210)*sh_7_11*y + math.sqrt(85)*sh_7_13*y + math.sqrt(2210)*sh_7_2*z + math.sqrt(2210)*sh_7_3*x)
    sh_8_4 = (1/40)*math.sqrt(935)*sh_7_10*y + (1/40)*math.sqrt(85)*sh_7_12*y - 1/40*math.sqrt(85)*sh_7_2*x + (1/10)*math.sqrt(85)*sh_7_3*z + (1/40)*math.sqrt(935)*sh_7_4*x
    sh_8_5 = (1/48)*math.sqrt(2)*(math.sqrt(102)*sh_7_11*y - math.sqrt(102)*sh_7_3*x + math.sqrt(1122)*sh_7_4*z + math.sqrt(561)*sh_7_5*x + math.sqrt(561)*sh_7_9*y)
    sh_8_6 = (1/16)*math.sqrt(34)*sh_7_10*y - 1/16*math.sqrt(34)*sh_7_4*x + (1/4)*math.sqrt(17)*sh_7_5*z + (1/16)*math.sqrt(102)*sh_7_6*x + (1/16)*math.sqrt(102)*sh_7_8*y
    sh_8_7 = -1/80*math.sqrt(1190)*sh_7_5*x + (1/40)*math.sqrt(1785)*sh_7_6*z + (1/20)*math.sqrt(255)*sh_7_7*y + (1/80)*math.sqrt(1190)*sh_7_9*y
    sh_8_8 = -1/60*math.sqrt(1785)*sh_7_6*y + (1/15)*math.sqrt(255)*sh_7_7*z - 1/60*math.sqrt(1785)*sh_7_8*x
    sh_8_9 = -1/80*math.sqrt(1190)*sh_7_5*y + (1/20)*math.sqrt(255)*sh_7_7*x + (1/40)*math.sqrt(1785)*sh_7_8*z - 1/80*math.sqrt(1190)*sh_7_9*x
    sh_8_10 = -1/16*math.sqrt(34)*sh_7_10*x - 1/16*math.sqrt(34)*sh_7_4*y - 1/16*math.sqrt(102)*sh_7_6*y + (1/16)*math.sqrt(102)*sh_7_8*x + (1/4)*math.sqrt(17)*sh_7_9*z
    sh_8_11 = (1/48)*math.sqrt(2)*(math.sqrt(1122)*sh_7_10*z - math.sqrt(102)*sh_7_11*x - math.sqrt(102)*sh_7_3*y - math.sqrt(561)*sh_7_5*y + math.sqrt(561)*sh_7_9*x)
    sh_8_12 = (1/40)*math.sqrt(935)*sh_7_10*x + (1/10)*math.sqrt(85)*sh_7_11*z - 1/40*math.sqrt(85)*sh_7_12*x - 1/40*math.sqrt(85)*sh_7_2*y - 1/40*math.sqrt(935)*sh_7_4*y
    sh_8_13 = (1/80)*math.sqrt(2)*(-math.sqrt(85)*sh_7_1*y + math.sqrt(2210)*sh_7_11*x + math.sqrt(2210)*sh_7_12*z - math.sqrt(85)*sh_7_13*x - math.sqrt(2210)*sh_7_3*y)
    sh_8_14 = -1/240*math.sqrt(510)*sh_7_0*y + (1/240)*math.sqrt(46410)*sh_7_12*x + (1/60)*math.sqrt(1785)*sh_7_13*z - 1/240*math.sqrt(510)*sh_7_14*x - 1/240*math.sqrt(46410)*sh_7_2*y
    sh_8_15 = -1/16*math.sqrt(238)*sh_7_1*y + (1/16)*math.sqrt(238)*sh_7_13*x + (1/8)*math.sqrt(17)*sh_7_14*z
    sh_8_16 = (1/4)*math.sqrt(17)*(-sh_7_0*y + sh_7_14*x)
    if l_max == 8:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
            sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16
        ], dim=dim)

    sh_9_0 = (1/6)*math.sqrt(38)*(sh_8_0*x + sh_8_16*y)
    sh_9_1 = (1/9)*math.sqrt(19)*(sh_8_0*z + 2*sh_8_1*x + 2*sh_8_15*y)
    sh_9_2 = -1/306*math.sqrt(646)*sh_8_0*x + (4/153)*math.sqrt(646)*sh_8_1*z + (2/153)*math.sqrt(4845)*sh_8_14*y + (1/306)*math.sqrt(646)*sh_8_16*y + (2/153)*math.sqrt(4845)*sh_8_2*x
    sh_9_3 = -1/306*math.sqrt(1938)*sh_8_1*x + (1/306)*math.sqrt(67830)*sh_8_13*y + (1/306)*math.sqrt(1938)*sh_8_15*y + (1/51)*math.sqrt(1615)*sh_8_2*z + (1/306)*math.sqrt(67830)*sh_8_3*x
    sh_9_4 = (1/306)*math.sqrt(58786)*sh_8_12*y + (1/153)*math.sqrt(969)*sh_8_14*y - 1/153*math.sqrt(969)*sh_8_2*x + (2/153)*math.sqrt(4522)*sh_8_3*z + (1/306)*math.sqrt(58786)*sh_8_4*x
    sh_9_5 = (1/153)*math.sqrt(12597)*sh_8_11*y + (1/153)*math.sqrt(1615)*sh_8_13*y - 1/153*math.sqrt(1615)*sh_8_3*x + (1/153)*math.sqrt(20995)*sh_8_4*z + (1/153)*math.sqrt(12597)*sh_8_5*x
    sh_9_6 = (1/153)*math.sqrt(10659)*sh_8_10*y + (1/306)*math.sqrt(9690)*sh_8_12*y - 1/306*math.sqrt(9690)*sh_8_4*x + (2/51)*math.sqrt(646)*sh_8_5*z + (1/153)*math.sqrt(10659)*sh_8_6*x
    sh_9_7 = (1/306)*math.sqrt(13566)*sh_8_11*y - 1/306*math.sqrt(13566)*sh_8_5*x + (1/153)*math.sqrt(24871)*sh_8_6*z + (1/306)*math.sqrt(35530)*sh_8_7*x + (1/306)*math.sqrt(35530)*sh_8_9*y
    sh_9_8 = (1/153)*math.sqrt(4522)*sh_8_10*y - 1/153*math.sqrt(4522)*sh_8_6*x + (4/153)*math.sqrt(1615)*sh_8_7*z + (1/51)*math.sqrt(1615)*sh_8_8*y
    sh_9_9 = (1/51)*math.sqrt(323)*(-2*sh_8_7*y + 3*sh_8_8*z - 2*sh_8_9*x)
    sh_9_10 = -1/153*math.sqrt(4522)*sh_8_10*x - 1/153*math.sqrt(4522)*sh_8_6*y + (1/51)*math.sqrt(1615)*sh_8_8*x + (4/153)*math.sqrt(1615)*sh_8_9*z
    sh_9_11 = (1/153)*math.sqrt(24871)*sh_8_10*z - 1/306*math.sqrt(13566)*sh_8_11*x - 1/306*math.sqrt(13566)*sh_8_5*y - 1/306*math.sqrt(35530)*sh_8_7*y + (1/306)*math.sqrt(35530)*sh_8_9*x
    sh_9_12 = (1/153)*math.sqrt(10659)*sh_8_10*x + (2/51)*math.sqrt(646)*sh_8_11*z - 1/306*math.sqrt(9690)*sh_8_12*x - 1/306*math.sqrt(9690)*sh_8_4*y - 1/153*math.sqrt(10659)*sh_8_6*y
    sh_9_13 = (1/153)*math.sqrt(12597)*sh_8_11*x + (1/153)*math.sqrt(20995)*sh_8_12*z - 1/153*math.sqrt(1615)*sh_8_13*x - 1/153*math.sqrt(1615)*sh_8_3*y - 1/153*math.sqrt(12597)*sh_8_5*y
    sh_9_14 = (1/306)*math.sqrt(58786)*sh_8_12*x + (2/153)*math.sqrt(4522)*sh_8_13*z - 1/153*math.sqrt(969)*sh_8_14*x - 1/153*math.sqrt(969)*sh_8_2*y - 1/306*math.sqrt(58786)*sh_8_4*y
    sh_9_15 = -1/306*math.sqrt(1938)*sh_8_1*y + (1/306)*math.sqrt(67830)*sh_8_13*x + (1/51)*math.sqrt(1615)*sh_8_14*z - 1/306*math.sqrt(1938)*sh_8_15*x - 1/306*math.sqrt(67830)*sh_8_3*y
    sh_9_16 = -1/306*math.sqrt(646)*sh_8_0*y + (2/153)*math.sqrt(4845)*sh_8_14*x + (4/153)*math.sqrt(646)*sh_8_15*z - 1/306*math.sqrt(646)*sh_8_16*x - 2/153*math.sqrt(4845)*sh_8_2*y
    sh_9_17 = (1/9)*math.sqrt(19)*(-2*sh_8_1*y + 2*sh_8_15*x + sh_8_16*z)
    sh_9_18 = (1/6)*math.sqrt(38)*(-sh_8_0*y + sh_8_16*x)
    if l_max == 9:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
            sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
            sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18
        ], dim=dim)

    sh_10_0 = (1/10)*math.sqrt(105)*(sh_9_0*x + sh_9_18*y)
    sh_10_1 = (1/10)*math.sqrt(21)*sh_9_0*z + (3/20)*math.sqrt(42)*sh_9_1*x + (3/20)*math.sqrt(42)*sh_9_17*y
    sh_10_2 = -1/380*math.sqrt(798)*sh_9_0*x + (3/95)*math.sqrt(399)*sh_9_1*z + (3/380)*math.sqrt(13566)*sh_9_16*y + (1/380)*math.sqrt(798)*sh_9_18*y + (3/380)*math.sqrt(13566)*sh_9_2*x
    sh_10_3 = -3/380*math.sqrt(266)*sh_9_1*x + (1/95)*math.sqrt(6783)*sh_9_15*y + (3/380)*math.sqrt(266)*sh_9_17*y + (3/190)*math.sqrt(2261)*sh_9_2*z + (1/95)*math.sqrt(6783)*sh_9_3*x
    sh_10_4 = (3/95)*math.sqrt(665)*sh_9_14*y + (3/190)*math.sqrt(133)*sh_9_16*y - 3/190*math.sqrt(133)*sh_9_2*x + (4/95)*math.sqrt(399)*sh_9_3*z + (3/95)*math.sqrt(665)*sh_9_4*x
    sh_10_5 = (21/380)*math.sqrt(190)*sh_9_13*y + (1/190)*math.sqrt(1995)*sh_9_15*y - 1/190*math.sqrt(1995)*sh_9_3*x + (3/38)*math.sqrt(133)*sh_9_4*z + (21/380)*math.sqrt(190)*sh_9_5*x
    sh_10_6 = (7/380)*math.sqrt(1482)*sh_9_12*y + (3/380)*math.sqrt(1330)*sh_9_14*y - 3/380*math.sqrt(1330)*sh_9_4*x + (21/95)*math.sqrt(19)*sh_9_5*z + (7/380)*math.sqrt(1482)*sh_9_6*x
    sh_10_7 = (3/190)*math.sqrt(1729)*sh_9_11*y + (21/380)*math.sqrt(38)*sh_9_13*y - 21/380*math.sqrt(38)*sh_9_5*x + (7/190)*math.sqrt(741)*sh_9_6*z + (3/190)*math.sqrt(1729)*sh_9_7*x
    sh_10_8 = (3/190)*math.sqrt(1463)*sh_9_10*y + (7/190)*math.sqrt(114)*sh_9_12*y - 7/190*math.sqrt(114)*sh_9_6*x + (6/95)*math.sqrt(266)*sh_9_7*z + (3/190)*math.sqrt(1463)*sh_9_8*x
    sh_10_9 = (3/190)*math.sqrt(798)*sh_9_11*y - 3/190*math.sqrt(798)*sh_9_7*x + (3/190)*math.sqrt(4389)*sh_9_8*z + (1/190)*math.sqrt(21945)*sh_9_9*y
    sh_10_10 = -3/190*math.sqrt(1995)*sh_9_10*x - 3/190*math.sqrt(1995)*sh_9_8*y + (1/19)*math.sqrt(399)*sh_9_9*z
    sh_10_11 = (3/190)*math.sqrt(4389)*sh_9_10*z - 3/190*math.sqrt(798)*sh_9_11*x - 3/190*math.sqrt(798)*sh_9_7*y + (1/190)*math.sqrt(21945)*sh_9_9*x
    sh_10_12 = (3/190)*math.sqrt(1463)*sh_9_10*x + (6/95)*math.sqrt(266)*sh_9_11*z - 7/190*math.sqrt(114)*sh_9_12*x - 7/190*math.sqrt(114)*sh_9_6*y - 3/190*math.sqrt(1463)*sh_9_8*y
    sh_10_13 = (3/190)*math.sqrt(1729)*sh_9_11*x + (7/190)*math.sqrt(741)*sh_9_12*z - 21/380*math.sqrt(38)*sh_9_13*x - 21/380*math.sqrt(38)*sh_9_5*y - 3/190*math.sqrt(1729)*sh_9_7*y
    sh_10_14 = (7/380)*math.sqrt(1482)*sh_9_12*x + (21/95)*math.sqrt(19)*sh_9_13*z - 3/380*math.sqrt(1330)*sh_9_14*x - 3/380*math.sqrt(1330)*sh_9_4*y - 7/380*math.sqrt(1482)*sh_9_6*y
    sh_10_15 = (21/380)*math.sqrt(190)*sh_9_13*x + (3/38)*math.sqrt(133)*sh_9_14*z - 1/190*math.sqrt(1995)*sh_9_15*x - 1/190*math.sqrt(1995)*sh_9_3*y - 21/380*math.sqrt(190)*sh_9_5*y
    sh_10_16 = (3/95)*math.sqrt(665)*sh_9_14*x + (4/95)*math.sqrt(399)*sh_9_15*z - 3/190*math.sqrt(133)*sh_9_16*x - 3/190*math.sqrt(133)*sh_9_2*y - 3/95*math.sqrt(665)*sh_9_4*y
    sh_10_17 = -3/380*math.sqrt(266)*sh_9_1*y + (1/95)*math.sqrt(6783)*sh_9_15*x + (3/190)*math.sqrt(2261)*sh_9_16*z - 3/380*math.sqrt(266)*sh_9_17*x - 1/95*math.sqrt(6783)*sh_9_3*y
    sh_10_18 = -1/380*math.sqrt(798)*sh_9_0*y + (3/380)*math.sqrt(13566)*sh_9_16*x + (3/95)*math.sqrt(399)*sh_9_17*z - 1/380*math.sqrt(798)*sh_9_18*x - 3/380*math.sqrt(13566)*sh_9_2*y
    sh_10_19 = -3/20*math.sqrt(42)*sh_9_1*y + (3/20)*math.sqrt(42)*sh_9_17*x + (1/10)*math.sqrt(21)*sh_9_18*z
    sh_10_20 = (1/10)*math.sqrt(105)*(-sh_9_0*y + sh_9_18*x)
    if l_max == 10:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
            sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
            sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18,
            sh_10_0, sh_10_1, sh_10_2, sh_10_3, sh_10_4, sh_10_5, sh_10_6, sh_10_7, sh_10_8, sh_10_9, sh_10_10, sh_10_11, sh_10_12, sh_10_13, sh_10_14, sh_10_15, sh_10_16, sh_10_17, sh_10_18, sh_10_19, sh_10_20
        ], dim=dim)

    sh_11_0 = (1/22)*math.sqrt(506)*(sh_10_0*x + sh_10_20*y)
    sh_11_1 = (1/11)*math.sqrt(23)*sh_10_0*z + (1/11)*math.sqrt(115)*sh_10_1*x + (1/11)*math.sqrt(115)*sh_10_19*y
    sh_11_2 = -1/462*math.sqrt(966)*sh_10_0*x + (2/231)*math.sqrt(4830)*sh_10_1*z + (1/231)*math.sqrt(45885)*sh_10_18*y + (1/231)*math.sqrt(45885)*sh_10_2*x + (1/462)*math.sqrt(966)*sh_10_20*y
    sh_11_3 = -1/154*math.sqrt(322)*sh_10_1*x + (1/154)*math.sqrt(18354)*sh_10_17*y + (1/154)*math.sqrt(322)*sh_10_19*y + (1/77)*math.sqrt(3059)*sh_10_2*z + (1/154)*math.sqrt(18354)*sh_10_3*x
    sh_11_4 = (1/154)*math.sqrt(16422)*sh_10_16*y + (1/77)*math.sqrt(161)*sh_10_18*y - 1/77*math.sqrt(161)*sh_10_2*x + (2/77)*math.sqrt(966)*sh_10_3*z + (1/154)*math.sqrt(16422)*sh_10_4*x
    sh_11_5 = (2/231)*math.sqrt(8211)*sh_10_15*y + (1/231)*math.sqrt(2415)*sh_10_17*y - 1/231*math.sqrt(2415)*sh_10_3*x + (1/231)*math.sqrt(41055)*sh_10_4*z + (2/231)*math.sqrt(8211)*sh_10_5*x
    sh_11_6 = (2/77)*math.sqrt(805)*sh_10_14*y + (1/154)*math.sqrt(1610)*sh_10_16*y - 1/154*math.sqrt(1610)*sh_10_4*x + (4/77)*math.sqrt(322)*sh_10_5*z + (2/77)*math.sqrt(805)*sh_10_6*x
    sh_11_7 = (1/22)*math.sqrt(230)*sh_10_13*y + (1/22)*math.sqrt(46)*sh_10_15*y - 1/22*math.sqrt(46)*sh_10_5*x + (1/11)*math.sqrt(115)*sh_10_6*z + (1/22)*math.sqrt(230)*sh_10_7*x
    sh_11_8 = (1/66)*math.sqrt(1794)*sh_10_12*y + (1/33)*math.sqrt(138)*sh_10_14*y - 1/33*math.sqrt(138)*sh_10_6*x + (4/33)*math.sqrt(69)*sh_10_7*z + (1/66)*math.sqrt(1794)*sh_10_8*x
    sh_11_9 = (1/77)*math.sqrt(2093)*sh_10_11*y + (1/77)*math.sqrt(966)*sh_10_13*y - 1/77*math.sqrt(966)*sh_10_7*x + (1/77)*math.sqrt(6279)*sh_10_8*z + (1/77)*math.sqrt(2093)*sh_10_9*x
    sh_11_10 = (1/77)*math.sqrt(3542)*sh_10_10*y + (1/154)*math.sqrt(4830)*sh_10_12*y - 1/154*math.sqrt(4830)*sh_10_8*x + (2/77)*math.sqrt(1610)*sh_10_9*z
    sh_11_11 = (1/21)*math.sqrt(483)*sh_10_10*z - 1/231*math.sqrt(26565)*sh_10_11*x - 1/231*math.sqrt(26565)*sh_10_9*y
    sh_11_12 = (1/77)*math.sqrt(3542)*sh_10_10*x + (2/77)*math.sqrt(1610)*sh_10_11*z - 1/154*math.sqrt(4830)*sh_10_12*x - 1/154*math.sqrt(4830)*sh_10_8*y
    sh_11_13 = (1/77)*math.sqrt(2093)*sh_10_11*x + (1/77)*math.sqrt(6279)*sh_10_12*z - 1/77*math.sqrt(966)*sh_10_13*x - 1/77*math.sqrt(966)*sh_10_7*y - 1/77*math.sqrt(2093)*sh_10_9*y
    sh_11_14 = (1/66)*math.sqrt(1794)*sh_10_12*x + (4/33)*math.sqrt(69)*sh_10_13*z - 1/33*math.sqrt(138)*sh_10_14*x - 1/33*math.sqrt(138)*sh_10_6*y - 1/66*math.sqrt(1794)*sh_10_8*y
    sh_11_15 = (1/22)*math.sqrt(230)*sh_10_13*x + (1/11)*math.sqrt(115)*sh_10_14*z - 1/22*math.sqrt(46)*sh_10_15*x - 1/22*math.sqrt(46)*sh_10_5*y - 1/22*math.sqrt(230)*sh_10_7*y
    sh_11_16 = (2/77)*math.sqrt(805)*sh_10_14*x + (4/77)*math.sqrt(322)*sh_10_15*z - 1/154*math.sqrt(1610)*sh_10_16*x - 1/154*math.sqrt(1610)*sh_10_4*y - 2/77*math.sqrt(805)*sh_10_6*y
    sh_11_17 = (2/231)*math.sqrt(8211)*sh_10_15*x + (1/231)*math.sqrt(41055)*sh_10_16*z - 1/231*math.sqrt(2415)*sh_10_17*x - 1/231*math.sqrt(2415)*sh_10_3*y - 2/231*math.sqrt(8211)*sh_10_5*y
    sh_11_18 = (1/154)*math.sqrt(16422)*sh_10_16*x + (2/77)*math.sqrt(966)*sh_10_17*z - 1/77*math.sqrt(161)*sh_10_18*x - 1/77*math.sqrt(161)*sh_10_2*y - 1/154*math.sqrt(16422)*sh_10_4*y
    sh_11_19 = -1/154*math.sqrt(322)*sh_10_1*y + (1/154)*math.sqrt(18354)*sh_10_17*x + (1/77)*math.sqrt(3059)*sh_10_18*z - 1/154*math.sqrt(322)*sh_10_19*x - 1/154*math.sqrt(18354)*sh_10_3*y
    sh_11_20 = -1/462*math.sqrt(966)*sh_10_0*y + (1/231)*math.sqrt(45885)*sh_10_18*x + (2/231)*math.sqrt(4830)*sh_10_19*z - 1/231*math.sqrt(45885)*sh_10_2*y - 1/462*math.sqrt(966)*sh_10_20*x
    sh_11_21 = -1/11*math.sqrt(115)*sh_10_1*y + (1/11)*math.sqrt(115)*sh_10_19*x + (1/11)*math.sqrt(23)*sh_10_20*z
    sh_11_22 = (1/22)*math.sqrt(506)*(-sh_10_0*y + sh_10_20*x)
    if l_max == 11:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
            sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
            sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18,
            sh_10_0, sh_10_1, sh_10_2, sh_10_3, sh_10_4, sh_10_5, sh_10_6, sh_10_7, sh_10_8, sh_10_9, sh_10_10, sh_10_11, sh_10_12, sh_10_13, sh_10_14, sh_10_15, sh_10_16, sh_10_17, sh_10_18, sh_10_19, sh_10_20,
            sh_11_0, sh_11_1, sh_11_2, sh_11_3, sh_11_4, sh_11_5, sh_11_6, sh_11_7, sh_11_8, sh_11_9, sh_11_10, sh_11_11, sh_11_12, sh_11_13, sh_11_14, sh_11_15, sh_11_16, sh_11_17, sh_11_18, sh_11_19, sh_11_20, sh_11_21, sh_11_22
        ], dim=dim)

    sh_12_0 = (5/12)*math.sqrt(6)*(sh_11_0*x + sh_11_22*y)
    sh_12_1 = (5/12)*sh_11_0*z + (5/24)*math.sqrt(22)*sh_11_1*x + (5/24)*math.sqrt(22)*sh_11_21*y
    sh_12_2 = -5/552*math.sqrt(46)*sh_11_0*x + (5/138)*math.sqrt(253)*sh_11_1*z + (5/552)*math.sqrt(10626)*sh_11_2*x + (5/552)*math.sqrt(10626)*sh_11_20*y + (5/552)*math.sqrt(46)*sh_11_22*y
    sh_12_3 = -5/552*math.sqrt(138)*sh_11_1*x + (5/276)*math.sqrt(2415)*sh_11_19*y + (5/92)*math.sqrt(161)*sh_11_2*z + (5/552)*math.sqrt(138)*sh_11_21*y + (5/276)*math.sqrt(2415)*sh_11_3*x
    sh_12_4 = (5/276)*math.sqrt(2185)*sh_11_18*y - 5/276*math.sqrt(69)*sh_11_2*x + (5/276)*math.sqrt(69)*sh_11_20*y + (5/69)*math.sqrt(115)*sh_11_3*z + (5/276)*math.sqrt(2185)*sh_11_4*x
    sh_12_5 = (5/184)*math.sqrt(874)*sh_11_17*y + (5/276)*math.sqrt(115)*sh_11_19*y - 5/276*math.sqrt(115)*sh_11_3*x + (5/276)*math.sqrt(2185)*sh_11_4*z + (5/184)*math.sqrt(874)*sh_11_5*x
    sh_12_6 = (5/552)*math.sqrt(3)*(math.sqrt(2346)*sh_11_16*y + math.sqrt(230)*sh_11_18*y - math.sqrt(230)*sh_11_4*x + 12*math.sqrt(23)*sh_11_5*z + math.sqrt(2346)*sh_11_6*x)
    sh_12_7 = (5/138)*math.sqrt(391)*sh_11_15*y + (5/552)*math.sqrt(966)*sh_11_17*y - 5/552*math.sqrt(966)*sh_11_5*x + (5/276)*math.sqrt(2737)*sh_11_6*z + (5/138)*math.sqrt(391)*sh_11_7*x
    sh_12_8 = (5/138)*math.sqrt(345)*sh_11_14*y + (5/276)*math.sqrt(322)*sh_11_16*y - 5/276*math.sqrt(322)*sh_11_6*x + (10/69)*math.sqrt(46)*sh_11_7*z + (5/138)*math.sqrt(345)*sh_11_8*x
    sh_12_9 = (5/552)*math.sqrt(4830)*sh_11_13*y + (5/92)*math.sqrt(46)*sh_11_15*y - 5/92*math.sqrt(46)*sh_11_7*x + (5/92)*math.sqrt(345)*sh_11_8*z + (5/552)*math.sqrt(4830)*sh_11_9*x
    sh_12_10 = (5/552)*math.sqrt(4186)*sh_11_10*x + (5/552)*math.sqrt(4186)*sh_11_12*y + (5/184)*math.sqrt(230)*sh_11_14*y - 5/184*math.sqrt(230)*sh_11_8*x + (5/138)*math.sqrt(805)*sh_11_9*z
    sh_12_11 = (5/276)*math.sqrt(3289)*sh_11_10*z + (5/276)*math.sqrt(1794)*sh_11_11*y + (5/552)*math.sqrt(2530)*sh_11_13*y - 5/552*math.sqrt(2530)*sh_11_9*x
    sh_12_12 = -5/276*math.sqrt(1518)*sh_11_10*y + (5/23)*math.sqrt(23)*sh_11_11*z - 5/276*math.sqrt(1518)*sh_11_12*x
    sh_12_13 = (5/276)*math.sqrt(1794)*sh_11_11*x + (5/276)*math.sqrt(3289)*sh_11_12*z - 5/552*math.sqrt(2530)*sh_11_13*x - 5/552*math.sqrt(2530)*sh_11_9*y
    sh_12_14 = -5/552*math.sqrt(4186)*sh_11_10*y + (5/552)*math.sqrt(4186)*sh_11_12*x + (5/138)*math.sqrt(805)*sh_11_13*z - 5/184*math.sqrt(230)*sh_11_14*x - 5/184*math.sqrt(230)*sh_11_8*y
    sh_12_15 = (5/552)*math.sqrt(4830)*sh_11_13*x + (5/92)*math.sqrt(345)*sh_11_14*z - 5/92*math.sqrt(46)*sh_11_15*x - 5/92*math.sqrt(46)*sh_11_7*y - 5/552*math.sqrt(4830)*sh_11_9*y
    sh_12_16 = (5/138)*math.sqrt(345)*sh_11_14*x + (10/69)*math.sqrt(46)*sh_11_15*z - 5/276*math.sqrt(322)*sh_11_16*x - 5/276*math.sqrt(322)*sh_11_6*y - 5/138*math.sqrt(345)*sh_11_8*y
    sh_12_17 = (5/138)*math.sqrt(391)*sh_11_15*x + (5/276)*math.sqrt(2737)*sh_11_16*z - 5/552*math.sqrt(966)*sh_11_17*x - 5/552*math.sqrt(966)*sh_11_5*y - 5/138*math.sqrt(391)*sh_11_7*y
    sh_12_18 = (5/552)*math.sqrt(3)*(math.sqrt(2346)*sh_11_16*x + 12*math.sqrt(23)*sh_11_17*z - math.sqrt(230)*sh_11_18*x - math.sqrt(230)*sh_11_4*y - math.sqrt(2346)*sh_11_6*y)
    sh_12_19 = (5/184)*math.sqrt(874)*sh_11_17*x + (5/276)*math.sqrt(2185)*sh_11_18*z - 5/276*math.sqrt(115)*sh_11_19*x - 5/276*math.sqrt(115)*sh_11_3*y - 5/184*math.sqrt(874)*sh_11_5*y
    sh_12_20 = (5/276)*math.sqrt(2185)*sh_11_18*x + (5/69)*math.sqrt(115)*sh_11_19*z - 5/276*math.sqrt(69)*sh_11_2*y - 5/276*math.sqrt(69)*sh_11_20*x - 5/276*math.sqrt(2185)*sh_11_4*y
    sh_12_21 = -5/552*math.sqrt(138)*sh_11_1*y + (5/276)*math.sqrt(2415)*sh_11_19*x + (5/92)*math.sqrt(161)*sh_11_20*z - 5/552*math.sqrt(138)*sh_11_21*x - 5/276*math.sqrt(2415)*sh_11_3*y
    sh_12_22 = -5/552*math.sqrt(46)*sh_11_0*y - 5/552*math.sqrt(10626)*sh_11_2*y + (5/552)*math.sqrt(10626)*sh_11_20*x + (5/138)*math.sqrt(253)*sh_11_21*z - 5/552*math.sqrt(46)*sh_11_22*x
    sh_12_23 = -5/24*math.sqrt(22)*sh_11_1*y + (5/24)*math.sqrt(22)*sh_11_21*x + (5/12)*sh_11_22*z
    sh_12_24 = (5/12)*math.sqrt(6)*(-sh_11_0*y + sh_11_22*x)
    if l_max == 12:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
            sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
            sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18,
            sh_10_0, sh_10_1, sh_10_2, sh_10_3, sh_10_4, sh_10_5, sh_10_6, sh_10_7, sh_10_8, sh_10_9, sh_10_10, sh_10_11, sh_10_12, sh_10_13, sh_10_14, sh_10_15, sh_10_16, sh_10_17, sh_10_18, sh_10_19, sh_10_20,
            sh_11_0, sh_11_1, sh_11_2, sh_11_3, sh_11_4, sh_11_5, sh_11_6, sh_11_7, sh_11_8, sh_11_9, sh_11_10, sh_11_11, sh_11_12, sh_11_13, sh_11_14, sh_11_15, sh_11_16, sh_11_17, sh_11_18, sh_11_19, sh_11_20, sh_11_21, sh_11_22,
            sh_12_0, sh_12_1, sh_12_2, sh_12_3, sh_12_4, sh_12_5, sh_12_6, sh_12_7, sh_12_8, sh_12_9, sh_12_10, sh_12_11, sh_12_12, sh_12_13, sh_12_14, sh_12_15, sh_12_16, sh_12_17, sh_12_18, sh_12_19, sh_12_20, sh_12_21, sh_12_22, sh_12_23, sh_12_24
        ], dim=dim)

    sh_13_0 = (3/26)*math.sqrt(78)*(sh_12_0*x + sh_12_24*y)
    sh_13_1 = (3/13)*math.sqrt(3)*sh_12_0*z + (9/13)*math.sqrt(2)*sh_12_1*x + (9/13)*math.sqrt(2)*sh_12_23*y
    sh_13_2 = -3/130*math.sqrt(6)*sh_12_0*x + (36/65)*sh_12_1*z + (9/65)*math.sqrt(46)*sh_12_2*x + (9/65)*math.sqrt(46)*sh_12_22*y + (3/130)*math.sqrt(6)*sh_12_24*y
    sh_13_3 = -9/130*math.sqrt(2)*sh_12_1*x + (9/65)*math.sqrt(23)*sh_12_2*z + (3/130)*math.sqrt(1518)*sh_12_21*y + (9/130)*math.sqrt(2)*sh_12_23*y + (3/130)*math.sqrt(1518)*sh_12_3*x
    sh_13_4 = -9/65*sh_12_2*x + (9/130)*math.sqrt(154)*sh_12_20*y + (9/65)*sh_12_22*y + (6/65)*math.sqrt(66)*sh_12_3*z + (9/130)*math.sqrt(154)*sh_12_4*x
    sh_13_5 = (9/65)*math.sqrt(35)*sh_12_19*y + (3/65)*math.sqrt(15)*sh_12_21*y - 3/65*math.sqrt(15)*sh_12_3*x + (9/65)*math.sqrt(35)*sh_12_4*z + (9/65)*math.sqrt(35)*sh_12_5*x
    sh_13_6 = (3/65)*math.sqrt(285)*sh_12_18*y + (9/130)*math.sqrt(10)*sh_12_20*y - 9/130*math.sqrt(10)*sh_12_4*x + (18/65)*math.sqrt(10)*sh_12_5*z + (3/65)*math.sqrt(285)*sh_12_6*x
    sh_13_7 = (9/130)*math.sqrt(114)*sh_12_17*y + (9/130)*math.sqrt(14)*sh_12_19*y - 9/130*math.sqrt(14)*sh_12_5*x + (3/65)*math.sqrt(399)*sh_12_6*z + (9/130)*math.sqrt(114)*sh_12_7*x
    sh_13_8 = (9/130)*math.sqrt(102)*sh_12_16*y + (3/65)*math.sqrt(42)*sh_12_18*y - 3/65*math.sqrt(42)*sh_12_6*x + (36/65)*math.sqrt(3)*sh_12_7*z + (9/130)*math.sqrt(102)*sh_12_8*x
    sh_13_9 = (6/65)*math.sqrt(51)*sh_12_15*y + (9/65)*math.sqrt(6)*sh_12_17*y - 9/65*math.sqrt(6)*sh_12_7*x + (9/65)*math.sqrt(51)*sh_12_8*z + (6/65)*math.sqrt(51)*sh_12_9*x
    sh_13_10 = (18/65)*math.sqrt(5)*sh_12_10*x + (18/65)*math.sqrt(5)*sh_12_14*y + (9/130)*math.sqrt(30)*sh_12_16*y - 9/130*math.sqrt(30)*sh_12_8*x + (12/65)*math.sqrt(30)*sh_12_9*z
    sh_13_11 = (9/65)*math.sqrt(55)*sh_12_10*z + (9/130)*math.sqrt(70)*sh_12_11*x + (9/130)*math.sqrt(70)*sh_12_13*y + (3/130)*math.sqrt(330)*sh_12_15*y - 3/130*math.sqrt(330)*sh_12_9*x
    sh_13_12 = -9/65*math.sqrt(11)*sh_12_10*x + (18/65)*math.sqrt(14)*sh_12_11*z + (3/65)*math.sqrt(273)*sh_12_12*y + (9/65)*math.sqrt(11)*sh_12_14*y
    sh_13_13 = -9/65*math.sqrt(26)*sh_12_11*y + (3/5)*math.sqrt(3)*sh_12_12*z - 9/65*math.sqrt(26)*sh_12_13*x
    sh_13_14 = -9/65*math.sqrt(11)*sh_12_10*y + (3/65)*math.sqrt(273)*sh_12_12*x + (18/65)*math.sqrt(14)*sh_12_13*z - 9/65*math.sqrt(11)*sh_12_14*x
    sh_13_15 = -9/130*math.sqrt(70)*sh_12_11*y + (9/130)*math.sqrt(70)*sh_12_13*x + (9/65)*math.sqrt(55)*sh_12_14*z - 3/130*math.sqrt(330)*sh_12_15*x - 3/130*math.sqrt(330)*sh_12_9*y
    sh_13_16 = -18/65*math.sqrt(5)*sh_12_10*y + (18/65)*math.sqrt(5)*sh_12_14*x + (12/65)*math.sqrt(30)*sh_12_15*z - 9/130*math.sqrt(30)*sh_12_16*x - 9/130*math.sqrt(30)*sh_12_8*y
    sh_13_17 = (6/65)*math.sqrt(51)*sh_12_15*x + (9/65)*math.sqrt(51)*sh_12_16*z - 9/65*math.sqrt(6)*sh_12_17*x - 9/65*math.sqrt(6)*sh_12_7*y - 6/65*math.sqrt(51)*sh_12_9*y
    sh_13_18 = (9/130)*math.sqrt(102)*sh_12_16*x + (36/65)*math.sqrt(3)*sh_12_17*z - 3/65*math.sqrt(42)*sh_12_18*x - 3/65*math.sqrt(42)*sh_12_6*y - 9/130*math.sqrt(102)*sh_12_8*y
    sh_13_19 = (9/130)*math.sqrt(114)*sh_12_17*x + (3/65)*math.sqrt(399)*sh_12_18*z - 9/130*math.sqrt(14)*sh_12_19*x - 9/130*math.sqrt(14)*sh_12_5*y - 9/130*math.sqrt(114)*sh_12_7*y
    sh_13_20 = (3/65)*math.sqrt(285)*sh_12_18*x + (18/65)*math.sqrt(10)*sh_12_19*z - 9/130*math.sqrt(10)*sh_12_20*x - 9/130*math.sqrt(10)*sh_12_4*y - 3/65*math.sqrt(285)*sh_12_6*y
    sh_13_21 = (9/65)*math.sqrt(35)*sh_12_19*x + (9/65)*math.sqrt(35)*sh_12_20*z - 3/65*math.sqrt(15)*sh_12_21*x - 3/65*math.sqrt(15)*sh_12_3*y - 9/65*math.sqrt(35)*sh_12_5*y
    sh_13_22 = -9/65*sh_12_2*y + (9/130)*math.sqrt(154)*sh_12_20*x + (6/65)*math.sqrt(66)*sh_12_21*z - 9/65*sh_12_22*x - 9/130*math.sqrt(154)*sh_12_4*y
    sh_13_23 = -9/130*math.sqrt(2)*sh_12_1*y + (3/130)*math.sqrt(1518)*sh_12_21*x + (9/65)*math.sqrt(23)*sh_12_22*z - 9/130*math.sqrt(2)*sh_12_23*x - 3/130*math.sqrt(1518)*sh_12_3*y
    sh_13_24 = -3/130*math.sqrt(6)*sh_12_0*y - 9/65*math.sqrt(46)*sh_12_2*y + (9/65)*math.sqrt(46)*sh_12_22*x + (36/65)*sh_12_23*z - 3/130*math.sqrt(6)*sh_12_24*x
    sh_13_25 = -9/13*math.sqrt(2)*sh_12_1*y + (9/13)*math.sqrt(2)*sh_12_23*x + (3/13)*math.sqrt(3)*sh_12_24*z
    sh_13_26 = (3/26)*math.sqrt(78)*(-sh_12_0*y + sh_12_24*x)
    if l_max == 13:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
            sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
            sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18,
            sh_10_0, sh_10_1, sh_10_2, sh_10_3, sh_10_4, sh_10_5, sh_10_6, sh_10_7, sh_10_8, sh_10_9, sh_10_10, sh_10_11, sh_10_12, sh_10_13, sh_10_14, sh_10_15, sh_10_16, sh_10_17, sh_10_18, sh_10_19, sh_10_20,
            sh_11_0, sh_11_1, sh_11_2, sh_11_3, sh_11_4, sh_11_5, sh_11_6, sh_11_7, sh_11_8, sh_11_9, sh_11_10, sh_11_11, sh_11_12, sh_11_13, sh_11_14, sh_11_15, sh_11_16, sh_11_17, sh_11_18, sh_11_19, sh_11_20, sh_11_21, sh_11_22,
            sh_12_0, sh_12_1, sh_12_2, sh_12_3, sh_12_4, sh_12_5, sh_12_6, sh_12_7, sh_12_8, sh_12_9, sh_12_10, sh_12_11, sh_12_12, sh_12_13, sh_12_14, sh_12_15, sh_12_16, sh_12_17, sh_12_18, sh_12_19, sh_12_20, sh_12_21, sh_12_22, sh_12_23, sh_12_24,
            sh_13_0, sh_13_1, sh_13_2, sh_13_3, sh_13_4, sh_13_5, sh_13_6, sh_13_7, sh_13_8, sh_13_9, sh_13_10, sh_13_11, sh_13_12, sh_13_13, sh_13_14, sh_13_15, sh_13_16, sh_13_17, sh_13_18, sh_13_19, sh_13_20, sh_13_21, sh_13_22, sh_13_23, sh_13_24, sh_13_25, sh_13_26
        ], dim=dim)

    sh_14_0 = (1/14)*math.sqrt(203)*(sh_13_0*x + sh_13_26*y)
    sh_14_1 = (1/14)*math.sqrt(29)*sh_13_0*z + (1/28)*math.sqrt(754)*sh_13_1*x + (1/28)*math.sqrt(754)*sh_13_25*y
    sh_14_2 = -1/252*math.sqrt(174)*sh_13_0*x + (1/63)*math.sqrt(1131)*sh_13_1*z + (5/252)*math.sqrt(2262)*sh_13_2*x + (5/252)*math.sqrt(2262)*sh_13_24*y + (1/252)*math.sqrt(174)*sh_13_26*y
    sh_14_3 = -1/84*math.sqrt(58)*sh_13_1*x + (5/42)*math.sqrt(29)*sh_13_2*z + (5/42)*math.sqrt(58)*sh_13_23*y + (1/84)*math.sqrt(58)*sh_13_25*y + (5/42)*math.sqrt(58)*sh_13_3*x
    sh_14_4 = -1/42*math.sqrt(29)*sh_13_2*x + (1/42)*math.sqrt(1334)*sh_13_22*y + (1/42)*math.sqrt(29)*sh_13_24*y + (2/21)*math.sqrt(58)*sh_13_3*z + (1/42)*math.sqrt(1334)*sh_13_4*x
    sh_14_5 = (1/1764)*math.sqrt(14)*(math.sqrt(154077)*sh_13_21*y + math.sqrt(6090)*sh_13_23*y - math.sqrt(6090)*sh_13_3*x + math.sqrt(140070)*sh_13_4*z + math.sqrt(154077)*sh_13_5*x)
    sh_14_6 = (1/84)*math.sqrt(4466)*sh_13_20*y + (1/84)*math.sqrt(290)*sh_13_22*y - 1/84*math.sqrt(290)*sh_13_4*x + (1/21)*math.sqrt(319)*sh_13_5*z + (1/84)*math.sqrt(4466)*sh_13_6*x
    sh_14_7 = (1/84)*math.sqrt(14)*(math.sqrt(290)*sh_13_19*y + math.sqrt(29)*sh_13_21*y - math.sqrt(29)*sh_13_5*x + math.sqrt(406)*sh_13_6*z + math.sqrt(290)*sh_13_7*x)
    sh_14_8 = (1/126)*math.sqrt(8265)*sh_13_18*y + (1/126)*math.sqrt(1218)*sh_13_20*y - 1/126*math.sqrt(1218)*sh_13_6*x + (2/63)*math.sqrt(870)*sh_13_7*z + (1/126)*math.sqrt(8265)*sh_13_8*x
    sh_14_9 = (1/84)*math.sqrt(3306)*sh_13_17*y + (1/42)*math.sqrt(174)*sh_13_19*y - 1/42*math.sqrt(174)*sh_13_7*x + (1/42)*math.sqrt(1653)*sh_13_8*z + (1/84)*math.sqrt(3306)*sh_13_9*x
    sh_14_10 = (1/84)*math.sqrt(2958)*sh_13_10*x + (1/84)*math.sqrt(2958)*sh_13_16*y + (1/84)*math.sqrt(870)*sh_13_18*y - 1/84*math.sqrt(870)*sh_13_8*x + (1/21)*math.sqrt(435)*sh_13_9*z
    sh_14_11 = (1/126)*math.sqrt(16269)*sh_13_10*z + (1/63)*math.sqrt(1479)*sh_13_11*x + (1/63)*math.sqrt(1479)*sh_13_15*y + (1/252)*math.sqrt(9570)*sh_13_17*y - 1/252*math.sqrt(9570)*sh_13_9*x
    sh_14_12 = -1/42*math.sqrt(319)*sh_13_10*x + (4/21)*math.sqrt(29)*sh_13_11*z + (1/21)*math.sqrt(145)*sh_13_12*x + (1/21)*math.sqrt(145)*sh_13_14*y + (1/42)*math.sqrt(319)*sh_13_16*y
    sh_14_13 = -1/42*math.sqrt(377)*sh_13_11*x + (1/42)*math.sqrt(1885)*sh_13_12*z + (1/42)*math.sqrt(1015)*sh_13_13*y + (1/42)*math.sqrt(377)*sh_13_15*y
    sh_14_14 = -1/126*math.sqrt(7917)*sh_13_12*y + (1/9)*math.sqrt(87)*sh_13_13*z - 1/126*math.sqrt(7917)*sh_13_14*x
    sh_14_15 = -1/42*math.sqrt(377)*sh_13_11*y + (1/42)*math.sqrt(1015)*sh_13_13*x + (1/42)*math.sqrt(1885)*sh_13_14*z - 1/42*math.sqrt(377)*sh_13_15*x
    sh_14_16 = -1/42*math.sqrt(319)*sh_13_10*y - 1/21*math.sqrt(145)*sh_13_12*y + (1/21)*math.sqrt(145)*sh_13_14*x + (4/21)*math.sqrt(29)*sh_13_15*z - 1/42*math.sqrt(319)*sh_13_16*x
    sh_14_17 = -1/63*math.sqrt(1479)*sh_13_11*y + (1/63)*math.sqrt(1479)*sh_13_15*x + (1/126)*math.sqrt(16269)*sh_13_16*z - 1/252*math.sqrt(9570)*sh_13_17*x - 1/252*math.sqrt(9570)*sh_13_9*y
    sh_14_18 = -1/84*math.sqrt(2958)*sh_13_10*y + (1/84)*math.sqrt(2958)*sh_13_16*x + (1/21)*math.sqrt(435)*sh_13_17*z - 1/84*math.sqrt(870)*sh_13_18*x - 1/84*math.sqrt(870)*sh_13_8*y
    sh_14_19 = (1/84)*math.sqrt(3306)*sh_13_17*x + (1/42)*math.sqrt(1653)*sh_13_18*z - 1/42*math.sqrt(174)*sh_13_19*x - 1/42*math.sqrt(174)*sh_13_7*y - 1/84*math.sqrt(3306)*sh_13_9*y
    sh_14_20 = (1/126)*math.sqrt(8265)*sh_13_18*x + (2/63)*math.sqrt(870)*sh_13_19*z - 1/126*math.sqrt(1218)*sh_13_20*x - 1/126*math.sqrt(1218)*sh_13_6*y - 1/126*math.sqrt(8265)*sh_13_8*y
    sh_14_21 = (1/84)*math.sqrt(14)*(math.sqrt(290)*sh_13_19*x + math.sqrt(406)*sh_13_20*z - math.sqrt(29)*sh_13_21*x - math.sqrt(29)*sh_13_5*y - math.sqrt(290)*sh_13_7*y)
    sh_14_22 = (1/84)*math.sqrt(4466)*sh_13_20*x + (1/21)*math.sqrt(319)*sh_13_21*z - 1/84*math.sqrt(290)*sh_13_22*x - 1/84*math.sqrt(290)*sh_13_4*y - 1/84*math.sqrt(4466)*sh_13_6*y
    sh_14_23 = (1/1764)*math.sqrt(14)*(math.sqrt(154077)*sh_13_21*x + math.sqrt(140070)*sh_13_22*z - math.sqrt(6090)*sh_13_23*x - math.sqrt(6090)*sh_13_3*y - math.sqrt(154077)*sh_13_5*y)
    sh_14_24 = -1/42*math.sqrt(29)*sh_13_2*y + (1/42)*math.sqrt(1334)*sh_13_22*x + (2/21)*math.sqrt(58)*sh_13_23*z - 1/42*math.sqrt(29)*sh_13_24*x - 1/42*math.sqrt(1334)*sh_13_4*y
    sh_14_25 = -1/84*math.sqrt(58)*sh_13_1*y + (5/42)*math.sqrt(58)*sh_13_23*x + (5/42)*math.sqrt(29)*sh_13_24*z - 1/84*math.sqrt(58)*sh_13_25*x - 5/42*math.sqrt(58)*sh_13_3*y
    sh_14_26 = -1/252*math.sqrt(174)*sh_13_0*y - 5/252*math.sqrt(2262)*sh_13_2*y + (5/252)*math.sqrt(2262)*sh_13_24*x + (1/63)*math.sqrt(1131)*sh_13_25*z - 1/252*math.sqrt(174)*sh_13_26*x
    sh_14_27 = -1/28*math.sqrt(754)*sh_13_1*y + (1/28)*math.sqrt(754)*sh_13_25*x + (1/14)*math.sqrt(29)*sh_13_26*z
    sh_14_28 = (1/14)*math.sqrt(203)*(-sh_13_0*y + sh_13_26*x)
    if l_max == 14:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
            sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
            sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18,
            sh_10_0, sh_10_1, sh_10_2, sh_10_3, sh_10_4, sh_10_5, sh_10_6, sh_10_7, sh_10_8, sh_10_9, sh_10_10, sh_10_11, sh_10_12, sh_10_13, sh_10_14, sh_10_15, sh_10_16, sh_10_17, sh_10_18, sh_10_19, sh_10_20,
            sh_11_0, sh_11_1, sh_11_2, sh_11_3, sh_11_4, sh_11_5, sh_11_6, sh_11_7, sh_11_8, sh_11_9, sh_11_10, sh_11_11, sh_11_12, sh_11_13, sh_11_14, sh_11_15, sh_11_16, sh_11_17, sh_11_18, sh_11_19, sh_11_20, sh_11_21, sh_11_22,
            sh_12_0, sh_12_1, sh_12_2, sh_12_3, sh_12_4, sh_12_5, sh_12_6, sh_12_7, sh_12_8, sh_12_9, sh_12_10, sh_12_11, sh_12_12, sh_12_13, sh_12_14, sh_12_15, sh_12_16, sh_12_17, sh_12_18, sh_12_19, sh_12_20, sh_12_21, sh_12_22, sh_12_23, sh_12_24,
            sh_13_0, sh_13_1, sh_13_2, sh_13_3, sh_13_4, sh_13_5, sh_13_6, sh_13_7, sh_13_8, sh_13_9, sh_13_10, sh_13_11, sh_13_12, sh_13_13, sh_13_14, sh_13_15, sh_13_16, sh_13_17, sh_13_18, sh_13_19, sh_13_20, sh_13_21, sh_13_22, sh_13_23, sh_13_24, sh_13_25, sh_13_26,
            sh_14_0, sh_14_1, sh_14_2, sh_14_3, sh_14_4, sh_14_5, sh_14_6, sh_14_7, sh_14_8, sh_14_9, sh_14_10, sh_14_11, sh_14_12, sh_14_13, sh_14_14, sh_14_15, sh_14_16, sh_14_17, sh_14_18, sh_14_19, sh_14_20, sh_14_21, sh_14_22, sh_14_23, sh_14_24, sh_14_25, sh_14_26, sh_14_27, sh_14_28
        ], dim=dim)

    sh_15_0 = (1/30)*math.sqrt(930)*(sh_14_0*x + sh_14_28*y)
    sh_15_1 = (1/15)*math.sqrt(31)*sh_14_0*z + (1/15)*math.sqrt(217)*sh_14_1*x + (1/15)*math.sqrt(217)*sh_14_27*y
    sh_15_2 = -1/870*math.sqrt(1798)*sh_14_0*x + (2/435)*math.sqrt(12586)*sh_14_1*z + (1/145)*math.sqrt(18879)*sh_14_2*x + (1/145)*math.sqrt(18879)*sh_14_26*y + (1/870)*math.sqrt(1798)*sh_14_28*y
    sh_15_3 = -1/870*math.sqrt(5394)*sh_14_1*x + (3/145)*math.sqrt(899)*sh_14_2*z + (1/290)*math.sqrt(70122)*sh_14_25*y + (1/870)*math.sqrt(5394)*sh_14_27*y + (1/290)*math.sqrt(70122)*sh_14_3*x
    sh_15_4 = -1/435*math.sqrt(2697)*sh_14_2*x + (1/174)*math.sqrt(23374)*sh_14_24*y + (1/435)*math.sqrt(2697)*sh_14_26*y + (2/435)*math.sqrt(23374)*sh_14_3*z + (1/174)*math.sqrt(23374)*sh_14_4*x
    sh_15_5 = (1/87)*math.sqrt(5394)*sh_14_23*y + (1/435)*math.sqrt(4495)*sh_14_25*y - 1/435*math.sqrt(4495)*sh_14_3*x + (1/87)*math.sqrt(4495)*sh_14_4*z + (1/87)*math.sqrt(5394)*sh_14_5*x
    sh_15_6 = (1/435)*math.sqrt(124062)*sh_14_22*y + (1/870)*math.sqrt(26970)*sh_14_24*y - 1/870*math.sqrt(26970)*sh_14_4*x + (4/145)*math.sqrt(899)*sh_14_5*z + (1/435)*math.sqrt(124062)*sh_14_6*x
    sh_15_7 = (1/870)*math.sqrt(454894)*sh_14_21*y + (1/870)*math.sqrt(37758)*sh_14_23*y - 1/870*math.sqrt(37758)*sh_14_5*x + (1/435)*math.sqrt(144739)*sh_14_6*z + (1/870)*math.sqrt(454894)*sh_14_7*x
    sh_15_8 = (1/870)*math.sqrt(415338)*sh_14_20*y + (1/435)*math.sqrt(12586)*sh_14_22*y - 1/435*math.sqrt(12586)*sh_14_6*x + (4/435)*math.sqrt(9889)*sh_14_7*z + (1/870)*math.sqrt(415338)*sh_14_8*x
    sh_15_9 = (1/435)*math.sqrt(94395)*sh_14_19*y + (1/145)*math.sqrt(1798)*sh_14_21*y - 1/145*math.sqrt(1798)*sh_14_7*x + (1/145)*math.sqrt(18879)*sh_14_8*z + (1/435)*math.sqrt(94395)*sh_14_9*x
    sh_15_10 = (1/435)*math.sqrt(85405)*sh_14_10*x + (1/435)*math.sqrt(85405)*sh_14_18*y + (1/290)*math.sqrt(8990)*sh_14_20*y - 1/290*math.sqrt(8990)*sh_14_8*x + (2/87)*math.sqrt(1798)*sh_14_9*z
    sh_15_11 = (1/435)*math.sqrt(187891)*sh_14_10*z + (1/290)*math.sqrt(34162)*sh_14_11*x + (1/290)*math.sqrt(34162)*sh_14_17*y + (1/870)*math.sqrt(98890)*sh_14_19*y - 1/870*math.sqrt(98890)*sh_14_9*x
    sh_15_12 = -1/435*math.sqrt(29667)*sh_14_10*x + (2/145)*math.sqrt(5394)*sh_14_11*z + (1/290)*math.sqrt(30566)*sh_14_12*x + (1/290)*math.sqrt(30566)*sh_14_16*y + (1/435)*math.sqrt(29667)*sh_14_18*y
    sh_15_13 = -1/435*math.sqrt(35061)*sh_14_11*x + (1/435)*math.sqrt(198679)*sh_14_12*z + (2/435)*math.sqrt(15283)*sh_14_13*x + (2/435)*math.sqrt(15283)*sh_14_15*y + (1/435)*math.sqrt(35061)*sh_14_17*y
    sh_15_14 = -1/870*math.sqrt(163618)*sh_14_12*x + (4/435)*math.sqrt(12586)*sh_14_13*z + (2/435)*math.sqrt(26970)*sh_14_14*y + (1/870)*math.sqrt(163618)*sh_14_16*y
    sh_15_15 = -1/435*math.sqrt(94395)*sh_14_13*y + (1/29)*math.sqrt(899)*sh_14_14*z - 1/435*math.sqrt(94395)*sh_14_15*x
    sh_15_16 = -1/870*math.sqrt(163618)*sh_14_12*y + (2/435)*math.sqrt(26970)*sh_14_14*x + (4/435)*math.sqrt(12586)*sh_14_15*z - 1/870*math.sqrt(163618)*sh_14_16*x
    sh_15_17 = -1/435*math.sqrt(35061)*sh_14_11*y - 2/435*math.sqrt(15283)*sh_14_13*y + (2/435)*math.sqrt(15283)*sh_14_15*x + (1/435)*math.sqrt(198679)*sh_14_16*z - 1/435*math.sqrt(35061)*sh_14_17*x
    sh_15_18 = -1/435*math.sqrt(29667)*sh_14_10*y - 1/290*math.sqrt(30566)*sh_14_12*y + (1/290)*math.sqrt(30566)*sh_14_16*x + (2/145)*math.sqrt(5394)*sh_14_17*z - 1/435*math.sqrt(29667)*sh_14_18*x
    sh_15_19 = -1/290*math.sqrt(34162)*sh_14_11*y + (1/290)*math.sqrt(34162)*sh_14_17*x + (1/435)*math.sqrt(187891)*sh_14_18*z - 1/870*math.sqrt(98890)*sh_14_19*x - 1/870*math.sqrt(98890)*sh_14_9*y
    sh_15_20 = -1/435*math.sqrt(85405)*sh_14_10*y + (1/435)*math.sqrt(85405)*sh_14_18*x + (2/87)*math.sqrt(1798)*sh_14_19*z - 1/290*math.sqrt(8990)*sh_14_20*x - 1/290*math.sqrt(8990)*sh_14_8*y
    sh_15_21 = (1/435)*math.sqrt(94395)*sh_14_19*x + (1/145)*math.sqrt(18879)*sh_14_20*z - 1/145*math.sqrt(1798)*sh_14_21*x - 1/145*math.sqrt(1798)*sh_14_7*y - 1/435*math.sqrt(94395)*sh_14_9*y
    sh_15_22 = (1/870)*math.sqrt(415338)*sh_14_20*x + (4/435)*math.sqrt(9889)*sh_14_21*z - 1/435*math.sqrt(12586)*sh_14_22*x - 1/435*math.sqrt(12586)*sh_14_6*y - 1/870*math.sqrt(415338)*sh_14_8*y
    sh_15_23 = (1/870)*math.sqrt(454894)*sh_14_21*x + (1/435)*math.sqrt(144739)*sh_14_22*z - 1/870*math.sqrt(37758)*sh_14_23*x - 1/870*math.sqrt(37758)*sh_14_5*y - 1/870*math.sqrt(454894)*sh_14_7*y
    sh_15_24 = (1/435)*math.sqrt(124062)*sh_14_22*x + (4/145)*math.sqrt(899)*sh_14_23*z - 1/870*math.sqrt(26970)*sh_14_24*x - 1/870*math.sqrt(26970)*sh_14_4*y - 1/435*math.sqrt(124062)*sh_14_6*y
    sh_15_25 = (1/87)*math.sqrt(5394)*sh_14_23*x + (1/87)*math.sqrt(4495)*sh_14_24*z - 1/435*math.sqrt(4495)*sh_14_25*x - 1/435*math.sqrt(4495)*sh_14_3*y - 1/87*math.sqrt(5394)*sh_14_5*y
    sh_15_26 = -1/435*math.sqrt(2697)*sh_14_2*y + (1/174)*math.sqrt(23374)*sh_14_24*x + (2/435)*math.sqrt(23374)*sh_14_25*z - 1/435*math.sqrt(2697)*sh_14_26*x - 1/174*math.sqrt(23374)*sh_14_4*y
    sh_15_27 = -1/870*math.sqrt(5394)*sh_14_1*y + (1/290)*math.sqrt(70122)*sh_14_25*x + (3/145)*math.sqrt(899)*sh_14_26*z - 1/870*math.sqrt(5394)*sh_14_27*x - 1/290*math.sqrt(70122)*sh_14_3*y
    sh_15_28 = -1/870*math.sqrt(1798)*sh_14_0*y - 1/145*math.sqrt(18879)*sh_14_2*y + (1/145)*math.sqrt(18879)*sh_14_26*x + (2/435)*math.sqrt(12586)*sh_14_27*z - 1/870*math.sqrt(1798)*sh_14_28*x
    sh_15_29 = -1/15*math.sqrt(217)*sh_14_1*y + (1/15)*math.sqrt(217)*sh_14_27*x + (1/15)*math.sqrt(31)*sh_14_28*z
    sh_15_30 = (1/30)*math.sqrt(930)*(-sh_14_0*y + sh_14_28*x)
    if l_max == 15:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
            sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
            sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18,
            sh_10_0, sh_10_1, sh_10_2, sh_10_3, sh_10_4, sh_10_5, sh_10_6, sh_10_7, sh_10_8, sh_10_9, sh_10_10, sh_10_11, sh_10_12, sh_10_13, sh_10_14, sh_10_15, sh_10_16, sh_10_17, sh_10_18, sh_10_19, sh_10_20,
            sh_11_0, sh_11_1, sh_11_2, sh_11_3, sh_11_4, sh_11_5, sh_11_6, sh_11_7, sh_11_8, sh_11_9, sh_11_10, sh_11_11, sh_11_12, sh_11_13, sh_11_14, sh_11_15, sh_11_16, sh_11_17, sh_11_18, sh_11_19, sh_11_20, sh_11_21, sh_11_22,
            sh_12_0, sh_12_1, sh_12_2, sh_12_3, sh_12_4, sh_12_5, sh_12_6, sh_12_7, sh_12_8, sh_12_9, sh_12_10, sh_12_11, sh_12_12, sh_12_13, sh_12_14, sh_12_15, sh_12_16, sh_12_17, sh_12_18, sh_12_19, sh_12_20, sh_12_21, sh_12_22, sh_12_23, sh_12_24,
            sh_13_0, sh_13_1, sh_13_2, sh_13_3, sh_13_4, sh_13_5, sh_13_6, sh_13_7, sh_13_8, sh_13_9, sh_13_10, sh_13_11, sh_13_12, sh_13_13, sh_13_14, sh_13_15, sh_13_16, sh_13_17, sh_13_18, sh_13_19, sh_13_20, sh_13_21, sh_13_22, sh_13_23, sh_13_24, sh_13_25, sh_13_26,
            sh_14_0, sh_14_1, sh_14_2, sh_14_3, sh_14_4, sh_14_5, sh_14_6, sh_14_7, sh_14_8, sh_14_9, sh_14_10, sh_14_11, sh_14_12, sh_14_13, sh_14_14, sh_14_15, sh_14_16, sh_14_17, sh_14_18, sh_14_19, sh_14_20, sh_14_21, sh_14_22, sh_14_23, sh_14_24, sh_14_25, sh_14_26, sh_14_27, sh_14_28,
            sh_15_0, sh_15_1, sh_15_2, sh_15_3, sh_15_4, sh_15_5, sh_15_6, sh_15_7, sh_15_8, sh_15_9, sh_15_10, sh_15_11, sh_15_12, sh_15_13, sh_15_14, sh_15_15, sh_15_16, sh_15_17, sh_15_18, sh_15_19, sh_15_20, sh_15_21, sh_15_22, sh_15_23, sh_15_24, sh_15_25, sh_15_26, sh_15_27, sh_15_28, sh_15_29, sh_15_30
        ], dim=dim)

    sh_16_0 = (1/8)*math.sqrt(66)*(sh_15_0*x + sh_15_30*y)
    sh_16_1 = (1/16)*math.sqrt(33)*sh_15_0*z + (3/32)*math.sqrt(110)*sh_15_1*x + (3/32)*math.sqrt(110)*sh_15_29*y
    sh_16_2 = -1/992*math.sqrt(2046)*sh_15_0*x + (3/248)*math.sqrt(1705)*sh_15_1*z + (3/992)*math.sqrt(98890)*sh_15_2*x + (3/992)*math.sqrt(98890)*sh_15_28*y + (1/992)*math.sqrt(2046)*sh_15_30*y
    sh_16_3 = -3/992*math.sqrt(682)*sh_15_1*x + (3/496)*math.sqrt(9889)*sh_15_2*z + (1/496)*math.sqrt(207669)*sh_15_27*y + (3/992)*math.sqrt(682)*sh_15_29*y + (1/496)*math.sqrt(207669)*sh_15_3*x
    sh_16_4 = -3/496*math.sqrt(341)*sh_15_2*x + (9/496)*math.sqrt(2387)*sh_15_26*y + (3/496)*math.sqrt(341)*sh_15_28*y + (1/124)*math.sqrt(7161)*sh_15_3*z + (9/496)*math.sqrt(2387)*sh_15_4*x
    sh_16_5 = (9/992)*math.sqrt(8866)*sh_15_25*y + (1/496)*math.sqrt(5115)*sh_15_27*y - 1/496*math.sqrt(5115)*sh_15_3*x + (9/496)*math.sqrt(1705)*sh_15_4*z + (9/992)*math.sqrt(8866)*sh_15_5*x
    sh_16_6 = (5/992)*math.sqrt(26598)*sh_15_24*y + (3/992)*math.sqrt(3410)*sh_15_26*y - 3/992*math.sqrt(3410)*sh_15_4*x + (3/248)*math.sqrt(4433)*sh_15_5*z + (5/992)*math.sqrt(26598)*sh_15_6*x
    sh_16_7 = (15/496)*math.sqrt(682)*sh_15_23*y + (3/992)*math.sqrt(4774)*sh_15_25*y - 3/992*math.sqrt(4774)*sh_15_5*x + (5/496)*math.sqrt(7161)*sh_15_6*z + (15/496)*math.sqrt(682)*sh_15_7*x
    sh_16_8 = (3/496)*math.sqrt(15686)*sh_15_22*y + (1/496)*math.sqrt(14322)*sh_15_24*y - 1/496*math.sqrt(14322)*sh_15_6*x + (3/62)*math.sqrt(341)*sh_15_7*z + (3/496)*math.sqrt(15686)*sh_15_8*x
    sh_16_9 = (11/992)*math.sqrt(4278)*sh_15_21*y + (3/496)*math.sqrt(2046)*sh_15_23*y - 3/496*math.sqrt(2046)*sh_15_7*x + (3/496)*math.sqrt(23529)*sh_15_8*z + (11/992)*math.sqrt(4278)*sh_15_9*x
    sh_16_10 = (33/992)*math.sqrt(434)*sh_15_10*x + (33/992)*math.sqrt(434)*sh_15_20*y + (3/992)*math.sqrt(10230)*sh_15_22*y - 3/992*math.sqrt(10230)*sh_15_8*x + (11/248)*math.sqrt(465)*sh_15_9*z
    sh_16_11 = (33/496)*math.sqrt(217)*sh_15_10*z + (3/496)*math.sqrt(11935)*sh_15_11*x + (3/496)*math.sqrt(11935)*sh_15_19*y + (11/992)*math.sqrt(930)*sh_15_21*y - 11/992*math.sqrt(930)*sh_15_9*x
    sh_16_12 = -33/496*math.sqrt(31)*sh_15_10*x + (3/124)*math.sqrt(1705)*sh_15_11*z + (1/496)*math.sqrt(97185)*sh_15_12*x + (1/496)*math.sqrt(97185)*sh_15_18*y + (33/496)*math.sqrt(31)*sh_15_20*y
    sh_16_13 = -3/496*math.sqrt(4433)*sh_15_11*x + (1/496)*math.sqrt(252681)*sh_15_12*z + (3/992)*math.sqrt(38874)*sh_15_13*x + (3/992)*math.sqrt(38874)*sh_15_17*y + (3/496)*math.sqrt(4433)*sh_15_19*y
    sh_16_14 = -1/992*math.sqrt(186186)*sh_15_12*x + (3/248)*math.sqrt(7161)*sh_15_13*z + (3/992)*math.sqrt(34782)*sh_15_14*x + (3/992)*math.sqrt(34782)*sh_15_16*y + (1/992)*math.sqrt(186186)*sh_15_18*y
    sh_16_15 = -3/992*math.sqrt(23870)*sh_15_13*x + (3/496)*math.sqrt(28985)*sh_15_14*z + (1/248)*math.sqrt(34782)*sh_15_15*y + (3/992)*math.sqrt(23870)*sh_15_17*y
    sh_16_16 = -3/248*math.sqrt(3410)*sh_15_14*y + (1/31)*math.sqrt(1023)*sh_15_15*z - 3/248*math.sqrt(3410)*sh_15_16*x
    sh_16_17 = -3/992*math.sqrt(23870)*sh_15_13*y + (1/248)*math.sqrt(34782)*sh_15_15*x + (3/496)*math.sqrt(28985)*sh_15_16*z - 3/992*math.sqrt(23870)*sh_15_17*x
    sh_16_18 = -1/992*math.sqrt(186186)*sh_15_12*y - 3/992*math.sqrt(34782)*sh_15_14*y + (3/992)*math.sqrt(34782)*sh_15_16*x + (3/248)*math.sqrt(7161)*sh_15_17*z - 1/992*math.sqrt(186186)*sh_15_18*x
    sh_16_19 = -3/496*math.sqrt(4433)*sh_15_11*y - 3/992*math.sqrt(38874)*sh_15_13*y + (3/992)*math.sqrt(38874)*sh_15_17*x + (1/496)*math.sqrt(252681)*sh_15_18*z - 3/496*math.sqrt(4433)*sh_15_19*x
    sh_16_20 = -33/496*math.sqrt(31)*sh_15_10*y - 1/496*math.sqrt(97185)*sh_15_12*y + (1/496)*math.sqrt(97185)*sh_15_18*x + (3/124)*math.sqrt(1705)*sh_15_19*z - 33/496*math.sqrt(31)*sh_15_20*x
    sh_16_21 = -3/496*math.sqrt(11935)*sh_15_11*y + (3/496)*math.sqrt(11935)*sh_15_19*x + (33/496)*math.sqrt(217)*sh_15_20*z - 11/992*math.sqrt(930)*sh_15_21*x - 11/992*math.sqrt(930)*sh_15_9*y
    sh_16_22 = -33/992*math.sqrt(434)*sh_15_10*y + (33/992)*math.sqrt(434)*sh_15_20*x + (11/248)*math.sqrt(465)*sh_15_21*z - 3/992*math.sqrt(10230)*sh_15_22*x - 3/992*math.sqrt(10230)*sh_15_8*y
    sh_16_23 = (11/992)*math.sqrt(4278)*sh_15_21*x + (3/496)*math.sqrt(23529)*sh_15_22*z - 3/496*math.sqrt(2046)*sh_15_23*x - 3/496*math.sqrt(2046)*sh_15_7*y - 11/992*math.sqrt(4278)*sh_15_9*y
    sh_16_24 = (3/496)*math.sqrt(15686)*sh_15_22*x + (3/62)*math.sqrt(341)*sh_15_23*z - 1/496*math.sqrt(14322)*sh_15_24*x - 1/496*math.sqrt(14322)*sh_15_6*y - 3/496*math.sqrt(15686)*sh_15_8*y
    sh_16_25 = (15/496)*math.sqrt(682)*sh_15_23*x + (5/496)*math.sqrt(7161)*sh_15_24*z - 3/992*math.sqrt(4774)*sh_15_25*x - 3/992*math.sqrt(4774)*sh_15_5*y - 15/496*math.sqrt(682)*sh_15_7*y
    sh_16_26 = (5/992)*math.sqrt(26598)*sh_15_24*x + (3/248)*math.sqrt(4433)*sh_15_25*z - 3/992*math.sqrt(3410)*sh_15_26*x - 3/992*math.sqrt(3410)*sh_15_4*y - 5/992*math.sqrt(26598)*sh_15_6*y
    sh_16_27 = (9/992)*math.sqrt(8866)*sh_15_25*x + (9/496)*math.sqrt(1705)*sh_15_26*z - 1/496*math.sqrt(5115)*sh_15_27*x - 1/496*math.sqrt(5115)*sh_15_3*y - 9/992*math.sqrt(8866)*sh_15_5*y
    sh_16_28 = -3/496*math.sqrt(341)*sh_15_2*y + (9/496)*math.sqrt(2387)*sh_15_26*x + (1/124)*math.sqrt(7161)*sh_15_27*z - 3/496*math.sqrt(341)*sh_15_28*x - 9/496*math.sqrt(2387)*sh_15_4*y
    sh_16_29 = -3/992*math.sqrt(682)*sh_15_1*y + (1/496)*math.sqrt(207669)*sh_15_27*x + (3/496)*math.sqrt(9889)*sh_15_28*z - 3/992*math.sqrt(682)*sh_15_29*x - 1/496*math.sqrt(207669)*sh_15_3*y
    sh_16_30 = -1/992*math.sqrt(2046)*sh_15_0*y - 3/992*math.sqrt(98890)*sh_15_2*y + (3/992)*math.sqrt(98890)*sh_15_28*x + (3/248)*math.sqrt(1705)*sh_15_29*z - 1/992*math.sqrt(2046)*sh_15_30*x
    sh_16_31 = -3/32*math.sqrt(110)*sh_15_1*y + (3/32)*math.sqrt(110)*sh_15_29*x + (1/16)*math.sqrt(33)*sh_15_30*z
    sh_16_32 = (1/8)*math.sqrt(66)*(-sh_15_0*y + sh_15_30*x)
    if l_max == 16:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
            sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
            sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18,
            sh_10_0, sh_10_1, sh_10_2, sh_10_3, sh_10_4, sh_10_5, sh_10_6, sh_10_7, sh_10_8, sh_10_9, sh_10_10, sh_10_11, sh_10_12, sh_10_13, sh_10_14, sh_10_15, sh_10_16, sh_10_17, sh_10_18, sh_10_19, sh_10_20,
            sh_11_0, sh_11_1, sh_11_2, sh_11_3, sh_11_4, sh_11_5, sh_11_6, sh_11_7, sh_11_8, sh_11_9, sh_11_10, sh_11_11, sh_11_12, sh_11_13, sh_11_14, sh_11_15, sh_11_16, sh_11_17, sh_11_18, sh_11_19, sh_11_20, sh_11_21, sh_11_22,
            sh_12_0, sh_12_1, sh_12_2, sh_12_3, sh_12_4, sh_12_5, sh_12_6, sh_12_7, sh_12_8, sh_12_9, sh_12_10, sh_12_11, sh_12_12, sh_12_13, sh_12_14, sh_12_15, sh_12_16, sh_12_17, sh_12_18, sh_12_19, sh_12_20, sh_12_21, sh_12_22, sh_12_23, sh_12_24,
            sh_13_0, sh_13_1, sh_13_2, sh_13_3, sh_13_4, sh_13_5, sh_13_6, sh_13_7, sh_13_8, sh_13_9, sh_13_10, sh_13_11, sh_13_12, sh_13_13, sh_13_14, sh_13_15, sh_13_16, sh_13_17, sh_13_18, sh_13_19, sh_13_20, sh_13_21, sh_13_22, sh_13_23, sh_13_24, sh_13_25, sh_13_26,
            sh_14_0, sh_14_1, sh_14_2, sh_14_3, sh_14_4, sh_14_5, sh_14_6, sh_14_7, sh_14_8, sh_14_9, sh_14_10, sh_14_11, sh_14_12, sh_14_13, sh_14_14, sh_14_15, sh_14_16, sh_14_17, sh_14_18, sh_14_19, sh_14_20, sh_14_21, sh_14_22, sh_14_23, sh_14_24, sh_14_25, sh_14_26, sh_14_27, sh_14_28,
            sh_15_0, sh_15_1, sh_15_2, sh_15_3, sh_15_4, sh_15_5, sh_15_6, sh_15_7, sh_15_8, sh_15_9, sh_15_10, sh_15_11, sh_15_12, sh_15_13, sh_15_14, sh_15_15, sh_15_16, sh_15_17, sh_15_18, sh_15_19, sh_15_20, sh_15_21, sh_15_22, sh_15_23, sh_15_24, sh_15_25, sh_15_26, sh_15_27, sh_15_28, sh_15_29, sh_15_30,
            sh_16_0, sh_16_1, sh_16_2, sh_16_3, sh_16_4, sh_16_5, sh_16_6, sh_16_7, sh_16_8, sh_16_9, sh_16_10, sh_16_11, sh_16_12, sh_16_13, sh_16_14, sh_16_15, sh_16_16, sh_16_17, sh_16_18, sh_16_19, sh_16_20, sh_16_21, sh_16_22, sh_16_23, sh_16_24, sh_16_25, sh_16_26, sh_16_27, sh_16_28, sh_16_29, sh_16_30, sh_16_31, sh_16_32
        ], dim=dim)

    sh_17_0 = (1/34)*math.sqrt(1190)*(sh_16_0*x + sh_16_32*y)
    sh_17_1 = (1/17)*math.sqrt(35)*sh_16_0*z + (2/17)*math.sqrt(70)*sh_16_1*x + (2/17)*math.sqrt(70)*sh_16_31*y
    sh_17_2 = -1/1122*math.sqrt(2310)*sh_16_0*x + (8/561)*math.sqrt(1155)*sh_16_1*z + (2/561)*math.sqrt(71610)*sh_16_2*x + (2/561)*math.sqrt(71610)*sh_16_30*y + (1/1122)*math.sqrt(2310)*sh_16_32*y
    sh_17_3 = -1/374*math.sqrt(770)*sh_16_1*x + (1/187)*math.sqrt(11935)*sh_16_2*z + (5/374)*math.sqrt(4774)*sh_16_29*y + (5/374)*math.sqrt(4774)*sh_16_3*x + (1/374)*math.sqrt(770)*sh_16_31*y
    sh_17_4 = -1/187*math.sqrt(385)*sh_16_2*x + (5/374)*math.sqrt(4466)*sh_16_28*y + (10/187)*math.sqrt(154)*sh_16_3*z + (1/187)*math.sqrt(385)*sh_16_30*y + (5/374)*math.sqrt(4466)*sh_16_4*x
    sh_17_5 = (7/561)*math.sqrt(4785)*sh_16_27*y + (5/561)*math.sqrt(231)*sh_16_29*y - 5/561*math.sqrt(231)*sh_16_3*x + (5/561)*math.sqrt(6699)*sh_16_4*z + (7/561)*math.sqrt(4785)*sh_16_5*x
    sh_17_6 = (21/187)*math.sqrt(55)*sh_16_26*y + (5/374)*math.sqrt(154)*sh_16_28*y - 5/374*math.sqrt(154)*sh_16_4*x + (14/187)*math.sqrt(110)*sh_16_5*z + (21/187)*math.sqrt(55)*sh_16_6*x
    sh_17_7 = (3/374)*math.sqrt(10010)*sh_16_25*y + (7/374)*math.sqrt(110)*sh_16_27*y - 7/374*math.sqrt(110)*sh_16_5*x + (21/187)*math.sqrt(55)*sh_16_6*z + (3/374)*math.sqrt(10010)*sh_16_7*x
    sh_17_8 = (5/1122)*math.sqrt(30030)*sh_16_24*y + (7/561)*math.sqrt(330)*sh_16_26*y - 7/561*math.sqrt(330)*sh_16_6*x + (4/561)*math.sqrt(15015)*sh_16_7*z + (5/1122)*math.sqrt(30030)*sh_16_8*x
    sh_17_9 = (5/187)*math.sqrt(770)*sh_16_23*y + (1/187)*math.sqrt(2310)*sh_16_25*y - 1/187*math.sqrt(2310)*sh_16_7*x + (5/187)*math.sqrt(1155)*sh_16_8*z + (5/187)*math.sqrt(770)*sh_16_9*x
    sh_17_10 = (1/187)*math.sqrt(17710)*sh_16_10*x + (1/187)*math.sqrt(17710)*sh_16_22*y + (5/374)*math.sqrt(462)*sh_16_24*y - 5/374*math.sqrt(462)*sh_16_8*x + (20/187)*math.sqrt(77)*sh_16_9*z
    sh_17_11 = (1/51)*math.sqrt(2415)*sh_16_10*z + (1/102)*math.sqrt(4830)*sh_16_11*x + (1/102)*math.sqrt(4830)*sh_16_21*y + (5/102)*math.sqrt(42)*sh_16_23*y - 5/102*math.sqrt(42)*sh_16_9*x
    sh_17_12 = -1/17*math.sqrt(35)*sh_16_10*x + (2/17)*math.sqrt(70)*sh_16_11*z + (7/34)*math.sqrt(10)*sh_16_12*x + (7/34)*math.sqrt(10)*sh_16_20*y + (1/17)*math.sqrt(35)*sh_16_22*y
    sh_17_13 = -1/187*math.sqrt(5005)*sh_16_11*x + (7/187)*math.sqrt(715)*sh_16_12*z + (35/187)*math.sqrt(11)*sh_16_13*x + (35/187)*math.sqrt(11)*sh_16_19*y + (1/187)*math.sqrt(5005)*sh_16_21*y
    sh_17_14 = -7/1122*math.sqrt(4290)*sh_16_12*x + (70/561)*math.sqrt(66)*sh_16_13*z + (5/561)*math.sqrt(4389)*sh_16_14*x + (5/561)*math.sqrt(4389)*sh_16_18*y + (7/1122)*math.sqrt(4290)*sh_16_20*y
    sh_17_15 = -35/374*math.sqrt(22)*sh_16_13*x + (5/187)*math.sqrt(1463)*sh_16_14*z + (1/374)*math.sqrt(43890)*sh_16_15*x + (1/374)*math.sqrt(43890)*sh_16_17*y + (35/374)*math.sqrt(22)*sh_16_19*y
    sh_17_16 = -10/187*math.sqrt(77)*sh_16_14*x + (4/187)*math.sqrt(2310)*sh_16_15*z + (1/187)*math.sqrt(19635)*sh_16_16*y + (10/187)*math.sqrt(77)*sh_16_18*y
    sh_17_17 = -2/561*math.sqrt(39270)*sh_16_15*y + (1/33)*math.sqrt(1155)*sh_16_16*z - 2/561*math.sqrt(39270)*sh_16_17*x
    sh_17_18 = -10/187*math.sqrt(77)*sh_16_14*y + (1/187)*math.sqrt(19635)*sh_16_16*x + (4/187)*math.sqrt(2310)*sh_16_17*z - 10/187*math.sqrt(77)*sh_16_18*x
    sh_17_19 = -35/374*math.sqrt(22)*sh_16_13*y - 1/374*math.sqrt(43890)*sh_16_15*y + (1/374)*math.sqrt(43890)*sh_16_17*x + (5/187)*math.sqrt(1463)*sh_16_18*z - 35/374*math.sqrt(22)*sh_16_19*x
    sh_17_20 = -7/1122*math.sqrt(4290)*sh_16_12*y - 5/561*math.sqrt(4389)*sh_16_14*y + (5/561)*math.sqrt(4389)*sh_16_18*x + (70/561)*math.sqrt(66)*sh_16_19*z - 7/1122*math.sqrt(4290)*sh_16_20*x
    sh_17_21 = -1/187*math.sqrt(5005)*sh_16_11*y - 35/187*math.sqrt(11)*sh_16_13*y + (35/187)*math.sqrt(11)*sh_16_19*x + (7/187)*math.sqrt(715)*sh_16_20*z - 1/187*math.sqrt(5005)*sh_16_21*x
    sh_17_22 = -1/17*math.sqrt(35)*sh_16_10*y - 7/34*math.sqrt(10)*sh_16_12*y + (7/34)*math.sqrt(10)*sh_16_20*x + (2/17)*math.sqrt(70)*sh_16_21*z - 1/17*math.sqrt(35)*sh_16_22*x
    sh_17_23 = -1/102*math.sqrt(4830)*sh_16_11*y + (1/102)*math.sqrt(4830)*sh_16_21*x + (1/51)*math.sqrt(2415)*sh_16_22*z - 5/102*math.sqrt(42)*sh_16_23*x - 5/102*math.sqrt(42)*sh_16_9*y
    sh_17_24 = -1/187*math.sqrt(17710)*sh_16_10*y + (1/187)*math.sqrt(17710)*sh_16_22*x + (20/187)*math.sqrt(77)*sh_16_23*z - 5/374*math.sqrt(462)*sh_16_24*x - 5/374*math.sqrt(462)*sh_16_8*y
    sh_17_25 = (5/187)*math.sqrt(770)*sh_16_23*x + (5/187)*math.sqrt(1155)*sh_16_24*z - 1/187*math.sqrt(2310)*sh_16_25*x - 1/187*math.sqrt(2310)*sh_16_7*y - 5/187*math.sqrt(770)*sh_16_9*y
    sh_17_26 = (5/1122)*math.sqrt(30030)*sh_16_24*x + (4/561)*math.sqrt(15015)*sh_16_25*z - 7/561*math.sqrt(330)*sh_16_26*x - 7/561*math.sqrt(330)*sh_16_6*y - 5/1122*math.sqrt(30030)*sh_16_8*y
    sh_17_27 = (3/374)*math.sqrt(10010)*sh_16_25*x + (21/187)*math.sqrt(55)*sh_16_26*z - 7/374*math.sqrt(110)*sh_16_27*x - 7/374*math.sqrt(110)*sh_16_5*y - 3/374*math.sqrt(10010)*sh_16_7*y
    sh_17_28 = (21/187)*math.sqrt(55)*sh_16_26*x + (14/187)*math.sqrt(110)*sh_16_27*z - 5/374*math.sqrt(154)*sh_16_28*x - 5/374*math.sqrt(154)*sh_16_4*y - 21/187*math.sqrt(55)*sh_16_6*y
    sh_17_29 = (7/561)*math.sqrt(4785)*sh_16_27*x + (5/561)*math.sqrt(6699)*sh_16_28*z - 5/561*math.sqrt(231)*sh_16_29*x - 5/561*math.sqrt(231)*sh_16_3*y - 7/561*math.sqrt(4785)*sh_16_5*y
    sh_17_30 = -1/187*math.sqrt(385)*sh_16_2*y + (5/374)*math.sqrt(4466)*sh_16_28*x + (10/187)*math.sqrt(154)*sh_16_29*z - 1/187*math.sqrt(385)*sh_16_30*x - 5/374*math.sqrt(4466)*sh_16_4*y
    sh_17_31 = -1/374*math.sqrt(770)*sh_16_1*y + (5/374)*math.sqrt(4774)*sh_16_29*x - 5/374*math.sqrt(4774)*sh_16_3*y + (1/187)*math.sqrt(11935)*sh_16_30*z - 1/374*math.sqrt(770)*sh_16_31*x
    sh_17_32 = -1/1122*math.sqrt(2310)*sh_16_0*y - 2/561*math.sqrt(71610)*sh_16_2*y + (2/561)*math.sqrt(71610)*sh_16_30*x + (8/561)*math.sqrt(1155)*sh_16_31*z - 1/1122*math.sqrt(2310)*sh_16_32*x
    sh_17_33 = -2/17*math.sqrt(70)*sh_16_1*y + (2/17)*math.sqrt(70)*sh_16_31*x + (1/17)*math.sqrt(35)*sh_16_32*z
    sh_17_34 = (1/34)*math.sqrt(1190)*(-sh_16_0*y + sh_16_32*x)
    if l_max == 17:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
            sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
            sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18,
            sh_10_0, sh_10_1, sh_10_2, sh_10_3, sh_10_4, sh_10_5, sh_10_6, sh_10_7, sh_10_8, sh_10_9, sh_10_10, sh_10_11, sh_10_12, sh_10_13, sh_10_14, sh_10_15, sh_10_16, sh_10_17, sh_10_18, sh_10_19, sh_10_20,
            sh_11_0, sh_11_1, sh_11_2, sh_11_3, sh_11_4, sh_11_5, sh_11_6, sh_11_7, sh_11_8, sh_11_9, sh_11_10, sh_11_11, sh_11_12, sh_11_13, sh_11_14, sh_11_15, sh_11_16, sh_11_17, sh_11_18, sh_11_19, sh_11_20, sh_11_21, sh_11_22,
            sh_12_0, sh_12_1, sh_12_2, sh_12_3, sh_12_4, sh_12_5, sh_12_6, sh_12_7, sh_12_8, sh_12_9, sh_12_10, sh_12_11, sh_12_12, sh_12_13, sh_12_14, sh_12_15, sh_12_16, sh_12_17, sh_12_18, sh_12_19, sh_12_20, sh_12_21, sh_12_22, sh_12_23, sh_12_24,
            sh_13_0, sh_13_1, sh_13_2, sh_13_3, sh_13_4, sh_13_5, sh_13_6, sh_13_7, sh_13_8, sh_13_9, sh_13_10, sh_13_11, sh_13_12, sh_13_13, sh_13_14, sh_13_15, sh_13_16, sh_13_17, sh_13_18, sh_13_19, sh_13_20, sh_13_21, sh_13_22, sh_13_23, sh_13_24, sh_13_25, sh_13_26,
            sh_14_0, sh_14_1, sh_14_2, sh_14_3, sh_14_4, sh_14_5, sh_14_6, sh_14_7, sh_14_8, sh_14_9, sh_14_10, sh_14_11, sh_14_12, sh_14_13, sh_14_14, sh_14_15, sh_14_16, sh_14_17, sh_14_18, sh_14_19, sh_14_20, sh_14_21, sh_14_22, sh_14_23, sh_14_24, sh_14_25, sh_14_26, sh_14_27, sh_14_28,
            sh_15_0, sh_15_1, sh_15_2, sh_15_3, sh_15_4, sh_15_5, sh_15_6, sh_15_7, sh_15_8, sh_15_9, sh_15_10, sh_15_11, sh_15_12, sh_15_13, sh_15_14, sh_15_15, sh_15_16, sh_15_17, sh_15_18, sh_15_19, sh_15_20, sh_15_21, sh_15_22, sh_15_23, sh_15_24, sh_15_25, sh_15_26, sh_15_27, sh_15_28, sh_15_29, sh_15_30,
            sh_16_0, sh_16_1, sh_16_2, sh_16_3, sh_16_4, sh_16_5, sh_16_6, sh_16_7, sh_16_8, sh_16_9, sh_16_10, sh_16_11, sh_16_12, sh_16_13, sh_16_14, sh_16_15, sh_16_16, sh_16_17, sh_16_18, sh_16_19, sh_16_20, sh_16_21, sh_16_22, sh_16_23, sh_16_24, sh_16_25, sh_16_26, sh_16_27, sh_16_28, sh_16_29, sh_16_30, sh_16_31, sh_16_32,
            sh_17_0, sh_17_1, sh_17_2, sh_17_3, sh_17_4, sh_17_5, sh_17_6, sh_17_7, sh_17_8, sh_17_9, sh_17_10, sh_17_11, sh_17_12, sh_17_13, sh_17_14, sh_17_15, sh_17_16, sh_17_17, sh_17_18, sh_17_19, sh_17_20, sh_17_21, sh_17_22, sh_17_23, sh_17_24, sh_17_25, sh_17_26, sh_17_27, sh_17_28, sh_17_29, sh_17_30, sh_17_31, sh_17_32, sh_17_33, sh_17_34
        ], dim=dim)
    raise ValueError(f'Max supported l_max is 17, got {l_max}')