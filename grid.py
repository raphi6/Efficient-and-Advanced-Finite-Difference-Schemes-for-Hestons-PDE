import math
import numpy as np
import scipy.sparse as sp


def make_grid(T, K, S_max, c, M1, V_max, M2):


    negative_discount = math.exp(-0.1 * T)
    positive_discount = math.exp(0.1 * T)

    S_lower_uniform = max(1/2, negative_discount) * K

    S_upper_uniform = max(1/2, positive_discount) * K


    xi_min = np.arcsinh(-S_lower_uniform / c)


    xi_interval = (S_upper_uniform - S_lower_uniform) / c


    xi_max = xi_interval + np.arcsinh((S_max - S_upper_uniform) / c)


    xi = np.linspace(xi_min, xi_max, M1)


    # xi in segments
    xi_first = [val for val in xi if val <= 0]
    xi_second = [val for val in xi if 0 < val < xi_interval]
    xi_third = [val for val in xi if val >= xi_interval]

    xi_first = np.array(xi_first)
    xi_second = np.array(xi_second)
    xi_third = np.array(xi_third)


    # s segments
    s_first = S_lower_uniform + c * np.sinh(xi_first)

    s_second = S_lower_uniform + c * xi_second


    s_third = S_upper_uniform + c * np.sinh(xi_third - xi_interval)


    # Combine segments
    s = np.concatenate([s_first, s_second, s_third])


    # v - direction

    d = (V_max / 500)

    psi_max = np.arcsinh(V_max / d)

    psi = np.linspace(0, psi_max, M2)

    v = d * np.sinh(psi)
    v[v < 0] = 0



    # Where s less than 2K
    index_left_plot = np.where(s < 2 * K)[0][-1]

    # Where v less than 1
    index_v_l1 = np.where(v < 1)[0][-1]

    index_s_boundary = np.arange(1, M1)
    index_v_boundary = np.arange(M2 - 1)

    index_s_boundary_size = index_s_boundary.size
    index_v_boundary_size = index_v_boundary.size

    # Create sparse diagonal matrices
    X_tilde = sp.diags(s, 0, shape=(M1, M1))
    Y_tilde = sp.diags(v, 0, shape=(M2, M2))

    # Extract submatrices by converting to dense format
    X_dense = X_tilde.toarray()
    Y_dense = Y_tilde.toarray()

    X = X_dense[index_s_boundary, :][:, index_s_boundary]
    Y = Y_dense[index_v_boundary, :][:, index_v_boundary]

    # Create grid for S and V
    S, V = np.meshgrid(s, v, indexing='ij')

    return S, V, s, v, index_left_plot, index_v_l1, index_s_boundary, index_v_boundary, index_s_boundary_size, index_v_boundary_size, X_tilde, Y_tilde, X, Y