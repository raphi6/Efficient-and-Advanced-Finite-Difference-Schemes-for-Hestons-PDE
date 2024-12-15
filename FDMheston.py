import QuantLib as ql
from datetime import datetime
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse import diags, eye, kron, csr_matrix, identity
import matplotlib.pyplot as plt
import math
import grid as grid
from scipy.sparse.linalg import splu
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import LinearNDInterpolator, griddata




def heston_price(spot, strike, maturity, risk_free_rate, dividend_yield, v0, kappa, theta, sigma, rho):
    """ Heston price via QuantLib """


    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    maturity_date = ql.Date().todaysDate() + ql.Period(int(maturity * 365), ql.Days)
    risk_free_rate_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(0, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(risk_free_rate)), ql.Actual365Fixed()))
    dividend_yield_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(0, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(dividend_yield)), ql.Actual365Fixed()))


    heston_process = ql.HestonProcess(risk_free_rate_handle, dividend_yield_handle, spot_handle, v0, kappa, theta,
                                      sigma, rho)


    heston_model = ql.HestonModel(heston_process)


    engine = ql.AnalyticHestonEngine(heston_model)


    european_option = ql.EuropeanOption(ql.PlainVanillaPayoff(ql.Option.Call, strike),
                                        ql.EuropeanExercise(maturity_date))
    european_option.setPricingEngine(engine)


    heston_price = european_option.NPV()

    return heston_price


def MCS(u, dt, LU1, LU2, A, A_0, prev, prev_0, prev_1, prev_2, next, next_0, next_1, next_2, theta):

    # Predictor step
    z0 = dt * (A @ u + prev)

    # Two correction steps
    z = LU1.solve(z0 + (theta * dt) * (next_1 - prev_1))
    z = LU2.solve(z + (theta * dt) * (next_2 - prev_2))

    # Predictor step
    z = z0 + (theta * dt) * (A_0 @ z + next_0 - prev_0) + ((1 / 2 - theta) * dt) * (A @ z + next - prev)

    # Two correction steps
    z = LU1.solve(z + (theta * dt) * (next_1 - prev_1))
    z = LU2.solve(z + (theta * dt) * (next_2 - prev_2))

    u = u + z

    return u

def CS(u, dt, LU1, LU2, A, A_0, prev, prev_0, prev_1, prev_2, next, next_0, next_1, next_2, theta):

    # Predictor step
    z0 = dt * (A @ u + prev)  # Y0 = Un-1 + dt * F(tn-1, Un-1)

    # Two corrector steps
    z = LU1.solve(z0 + (theta * dt) * (next_1 - prev_1))  # Y1
    z = LU2.solve(z + (theta * dt) * (next_2 - prev_2))  # Y2

    # Correction for mixed derivative term
    z_hat = z0 + 0.5 * dt * (A_0 @ z + next_0 - prev_0)

    # Two corrector steps
    z_tilde = LU1.solve(z_hat + (theta * dt) * (next_1 - prev_1))  # Y1
    z_tilde = LU2.solve(z_tilde + (theta * dt) * (next_2 - prev_2))  # Y2

    u = u + z_tilde

    return u



def Do(u, dt, LU1, LU2, A, prev, prev_1, next_1, prev_2, next_2, theta):

    # Explicit Predictor stage
    z0 = dt * (A @ u + prev)

    # Two Implicit Corrector stages
    z = LU1.solve(z0 + theta * dt * (next_1 - prev_1))
    z = LU2.solve(z + theta * dt * (next_2 - prev_2))

    u = u + z

    return u


def HV(u, dt, LU1, LU2, A, A_0, prev, prev_0, prev_1, prev_2, next, next_0, next_1, next_2, theta):

    # Step 1: Initial explicit step
    Y0 = u + dt * (A @ u + prev)

    # Step 2: Solving stages for Yj
    Z1 = Y0 - u
    Y1 = Z1 + theta * dt * (next_1 - prev_1)
    Y1 = LU1.solve(Y1)
    Y1 = LU2.solve(Y1)
    Y1 = Y1 + u

    Z2 = Y1 - u
    Y2 = Z2 + theta * dt * (next_2 - prev_2)
    Y2 = LU1.solve(Y2)
    Y2 = LU2.solve(Y2)
    Y2 = Y2 + u

    # Step 3: Corrector step
    Y0_hat = Y0 + (0.5 * dt) * (next_0 - prev_0)

    # Step 4: Solving stages for hat{Y}_j
    Z1_hat = Y0_hat - u
    hat_Y1 = Z1_hat + theta * dt * (next_1 - prev_1)
    hat_Y1 = LU1.solve(hat_Y1)
    hat_Y1 = LU2.solve(hat_Y1)
    hat_Y1 = hat_Y1 + u

    Z2_hat = hat_Y1 - Y2
    hat_Y2 = Z2_hat + theta * dt * (next_2 - prev_2)
    hat_Y2 = LU1.solve(hat_Y2)
    hat_Y2 = LU2.solve(hat_Y2)
    hat_Y2 = hat_Y2 + Y2

    # Step 5: Final update
    u = hat_Y2

    return u



def plot_dense_matrix(matrix, title):
    row, col = np.where(matrix != 0)
    plt.figure(figsize=(10, 10))
    plt.scatter(col, row, marker='o', color='blue', s=1)
    plt.gca().invert_yaxis()
    plt.xlabel('Column index')
    plt.ylabel('Row index')
    plt.title(title)
    plt.show()


def plot_sparse_matrix(matrix, title):
    row, col = matrix.nonzero()
    plt.figure(figsize=(10, 10))
    plt.scatter(col, row, marker='o', color='blue', s=1)
    plt.gca().invert_yaxis()
    plt.xlabel('Column index')
    plt.ylabel('Row index')
    plt.title(title)
    plt.show()



def FDMHeston(kappa = 0.9, eta = 0.04, sigma = 0.3, rho = 0.4, rd = 0.025, rf = 0.0, T = 1, K = 100, m1=150, N=500,
              scheme = 'HV', theta_exp=0.0, S_max_multiplier = 30, V_max_multiplier = 15):



    S_max = K * S_max_multiplier


    m1_tilde = m1 + 1

    V_max = 1 * V_max_multiplier

    m2 = round(0.5 * m1)

    m2_tilde = m2 + 1

    dt = T / N




    c = K / 10

    S, V, s, v, index_left_plot, index_v_l1, index_s_boundary, index_v_boundary, index_s_boundary_size, \
    index_v_boundary_size, X_tilde, Y_tilde, X_diags, Y_diags = grid.make_grid(T, K, S_max, c, m1_tilde, V_max, m2_tilde)


    u = np.maximum(S - K, 0)

    # Find  where s > K
    index_sK = np.argmax(s > K)

    right = s[index_sK] - K
    left = s[index_sK - 1] - K

    index_sK -= np.abs(right) > np.abs(left)

    # s_left and s_right
    s_left = (s[index_sK - 1] + s[index_sK]) / 2
    s_right = (s[index_sK] + s[index_sK + 1]) / 2

    # Update u at the index index_sK
    u[index_sK, :] = (1/2) * (s_right - K) ** 2 / (s_right - s_left)

    # Extract the submatrix)
    u_sub = u[np.ix_(index_s_boundary, index_v_boundary)]

    # Vectorise
    u_flat = u_sub.ravel(order='F')

    u = u_flat

    # Gamma_tilde matrix
    Gamma_tilde = np.full((m1_tilde, m2_tilde), 0)

    Gamma_tilde[:, - 1] = (s)

    start2 = datetime.now()

    # Discretisation Matrices - Starting with Coefficients
    offsets_central = [-1, 0, 1]
    offsets_backward = [-2, -1, 0]
    # Successive mesh widths in s-direction, and change in mesh widths
    Delta_s = np.diff(s[:m1_tilde])
    Delta_s_i = Delta_s[:-1]
    Delta_s_i_1 = Delta_s[1:]

    # Betas in s-direction
    beta_minus_s = -Delta_s_i_1 / (Delta_s_i * (Delta_s_i + Delta_s_i_1))
    beta_plus_s = Delta_s_i / (Delta_s_i_1 * (Delta_s_i + Delta_s_i_1))
    beta_zero_s = -(beta_minus_s + beta_plus_s)

    # detlas in s direction (2nd derivative)
    delta_minus_s = 2 / (Delta_s_i * (Delta_s_i + Delta_s_i_1))
    delta_plus_s = 2 / (Delta_s_i_1 * (Delta_s_i + Delta_s_i_1))
    delta_zero_s = -(delta_minus_s + delta_plus_s)

    # Pad the arrays to work in sparse matrix format
    beta_minus_s = np.append(beta_minus_s, [0, 0])
    beta_zero_s = np.concatenate(([0], beta_zero_s, [0]))
    beta_plus_s = np.concatenate(([0], beta_plus_s))

    delta_minus_s = np.append(delta_minus_s, [0, 0])
    delta_zero_s = np.concatenate(([0], delta_zero_s, [0]))
    delta_plus_s = np.concatenate(([0], delta_plus_s))


    # Now same thing but in v direction
    # Successive mesh widths in v-direction
    Delta_v = np.diff(v[:m2_tilde])
    Delta_v_i = Delta_v[:-1]
    Delta_v_i_1 = Delta_v[1:]

    # Beta's for v direction
    beta_minus_v = -Delta_v_i_1 / (Delta_v_i * (Delta_v_i + Delta_v_i_1))
    beta_plus_v = Delta_v_i / (Delta_v_i_1 * (Delta_v_i + Delta_v_i_1))
    beta_zero_v = -(beta_minus_v + beta_plus_v)

    # Alpha's for v direction
    alpha_minus_2_v = -(beta_minus_v)
    alpha_minus_v = -(Delta_v_i + Delta_v_i_1) / (Delta_v_i * Delta_v_i_1)
    alpha_zero_v = -(alpha_minus_2_v + alpha_minus_v)

    # delta's for v direction (2nd derivative)
    delta_minus_v = 2 / (Delta_v_i * (Delta_v_i + Delta_v_i_1))
    delta_plus_v = 2 / (Delta_v_i_1 * (Delta_v_i + Delta_v_i_1))
    delta_zero_v = -(delta_minus_v + delta_plus_v)


    # Pad the arrays to work in sparse matrix format using np.pad
    alpha_minus_2_v = np.pad(alpha_minus_2_v, (0, 2), mode='constant')
    alpha_minus_v = np.pad(alpha_minus_v, (1, 1), mode='constant')
    alpha_zero_v = np.pad(alpha_zero_v, (2, 0), mode='constant')

    beta_minus_v = np.pad(beta_minus_v, (0, 2), mode='constant')
    beta_zero_v = np.pad(beta_zero_v, (1, 1), mode='constant')
    beta_plus_v = np.pad(beta_plus_v, (1, 0), mode='constant')

    delta_minus_v = np.pad(delta_minus_v, (0, 2), mode='constant')
    delta_zero_v = np.pad(delta_zero_v, (1, 1), mode='constant')
    delta_plus_v = np.pad(delta_plus_v, (1, 0), mode='constant')



    # Discretisation matrices


    # e.g D_s_tilde is \widetilde{D}_S  - First derivative in S-direction - This follows central scheme
    inputs = [beta_minus_s, beta_zero_s, beta_plus_s]
    D_s_tilde = diags(inputs, offsets_central, shape=(m1_tilde, m1_tilde), format='lil')

    # Central scheme isn't followed at the upper boundary, This is implementing the Neumann boundary
    D_s_tilde[m1_tilde - 1, :] = 0


    #  For \widetilde{D}_v - First derivative in v direction - Here we have 3 schemes(Forward scheme at v=0,
    #                        Central scheme at between 0 and 1, Backward scheme for v larger than 1)

    # Central scheme so using Beta's on diagonals
    inputs = [beta_minus_v, beta_zero_v, beta_plus_v]
    D_v_central_tilde = diags(inputs, offsets_central, shape=(m2_tilde, m2_tilde))

    # Backward scheme with Alpha's on main diagonal and the two diagonals below
    inputs = [alpha_minus_2_v, alpha_minus_v, alpha_zero_v]
    D_v_backward_tilde = diags(inputs, offsets_backward, shape=(m2_tilde, m2_tilde))


    # \widetilde{D}_v  (almost all together)
    D_v_tilde = D_v_backward_tilde.tocsr()

    D_v_central_tilde = D_v_central_tilde.tocsr()

    # Now combine D_v_tilde (less than index_v_l1 being central scheme) and for bigger than index_v_l1 we have backward scheme
    D_v_tilde[:index_v_l1, :] = D_v_central_tilde[:index_v_l1, :]


    # Manually implementing forward scheme at the lower boundary, v=0. See section 5.3.3
    Delta_v_i_1 = Delta_v[0]
    Delta_v_i_2 = Delta_v[1]

    D_v_tilde[0, :] = 0
    D_v_tilde[0, 0] = ((-2 * Delta_v_i_1) - Delta_v_i_2) / (Delta_v_i_1 * (Delta_v_i_1 + Delta_v_i_2))   # gamma_0
    D_v_tilde[0, 1] = (Delta_v_i_1 + Delta_v_i_2) / (Delta_v_i_1 * Delta_v_i_2)                          # gamma 1
    D_v_tilde[0, 2] = -Delta_v_i_1 / (Delta_v_i_2 * (Delta_v_i_1 + Delta_v_i_2))                         # gamma 2


    # D_ss_tilde  Second derivative in S-direction
    inputs = [delta_minus_s, delta_zero_s, delta_plus_s]

    D_ss_tilde = diags(inputs, offsets_central, shape=(m1_tilde, m1_tilde), format='lil')


    # Neumann condition when i = m1_tilde
    D_ss_tilde[- 1, :] = 0

    Delta_s_i = Delta_s[m1 - 1]
    Delta_s_i_1 = 0                         # From Neumman = 0

    D_ss_tilde[- 1, - 2] = 2 / (Delta_s_i * (Delta_s_i + Delta_s_i_1))      # delta -1

    D_ss_tilde[- 1, - 1] = -2 / (Delta_s_i * (Delta_s_i + Delta_s_i_1))     # delta 0


    # D_vv_tilde
    inputs = [delta_minus_v, delta_zero_v, delta_plus_v]

    D_vv_tilde = diags(inputs, offsets_central, shape=(m2_tilde, m2_tilde))


    # D_s_mixed_tilde
    inputs = [beta_minus_s, beta_zero_s, beta_plus_s]

    D_s_mixed_tilde = diags(inputs, offsets_central, shape=(m1_tilde, m1_tilde))


    # D_v_mixed_tilde
    inputs = [beta_minus_v, beta_zero_v, beta_plus_v]

    D_v_mixed_tilde = diags(inputs, offsets_central, shape=(m2_tilde, m2_tilde))


    # Set the last row to 0 for D_s_mixed_tilde (see 5.3.3)
    D_s_mixed_tilde = D_s_mixed_tilde.tocsc()
    D_s_mixed_tilde[- 1, :] = 0

    D_vv_tilde = D_vv_tilde.tocsc()
    D_v_mixed_tilde = D_v_mixed_tilde.tocsc()



    # From tilde to no tilde
    D_s = D_s_tilde[index_s_boundary, :][:, index_s_boundary]

    D_v = D_v_tilde[index_v_boundary, :][:, index_v_boundary]

    D_ss = D_ss_tilde[index_s_boundary, :][:, index_s_boundary]

    D_vv = D_vv_tilde[index_v_boundary, :][:, index_v_boundary]

    D_s_mixed = D_s_mixed_tilde[index_s_boundary, :][:, index_s_boundary]

    D_v_mixed = D_v_mixed_tilde[index_v_boundary, :][:, index_v_boundary]

    # E_tilde matrix                    - Neumann condition, last row has 1s
    E_tilde = csr_matrix((m1_tilde, m2_tilde))

    # Set the last row to 1s
    E_tilde[-2 + 1, :] = np.ones((m2_tilde, 1))


    # Sparse identity matrix for S
    I_s = identity(index_s_boundary_size)
    # Sparse identity matrix for v
    I_v = identity(index_v_boundary_size)
    # Sparse identity matrix for v (extended)
    I_v_tilde = identity(m2_tilde)



    # A_0
    comp_1 = (rho * sigma) * Y_diags @ D_v_mixed
    comp_2 = X_diags @ D_s_mixed
    A_0 = kron(comp_1, comp_2)
    # plot_sparse_matrix(A_0, 'A_0')    nice plots (matches de Graaf)

    # A_1
    comp_1 = Y_diags
    comp_2 = 0.5 * X_diags @ X_diags @ D_ss
    comp_3 = I_v
    comp_4 = (rd) * X_diags @ D_s
    comp_5 = 0.5 * rd
    comp_6 = I_v
    comp_7 = I_s
    A_1 = kron(comp_1, comp_2) + kron(comp_3, comp_4) - comp_5 * kron(comp_6, comp_7)
    # plot_sparse_matrix(A_1, "A_1")

    # A_2
    comp_1 = (0.5 * sigma ** 2) * Y_diags @ D_vv
    comp_2 = kappa * (eta * I_v - Y_diags) @ D_v
    kron_comp = kron(comp_1 + comp_2, I_s)
    comp_3 = (0.5 * rd) * kron(I_v, I_s)
    A_2 = kron_comp - comp_3
    # plot_sparse_matrix(A_2, "A_2")

    # Finally A (all together)
    A = A_0 + A_1 + A_2

    #plot_sparse_matrix(A, "A")

    end2 = datetime.now()
    time2 = (end2 - start2).total_seconds()

    #print("Computation Time 2 (A matrix):", time2)



    # G matrix (boundaries)

    # g_0
    comp_1 = D_s_mixed_tilde @ Gamma_tilde
    comp_2 = X_tilde @ comp_1
    comp_3 = comp_2 @ D_v_mixed_tilde.transpose()
    comp_4 = comp_3 @ Y_tilde
    g_0 = (rho * sigma) * comp_4


    # g_1
    comp_1_1 = D_ss_tilde @ Gamma_tilde
    comp_1_2 = comp_1_1 + (2 / Delta_s[m1 - 1]) * E_tilde
    comp_1_3 = X_tilde @ X_tilde
    comp_1_4 = comp_1_3 @ comp_1_2
    comp_1_5 = comp_1_4 @ Y_tilde
    first_term = 0.5 * comp_1_5

    comp_2_1 = D_s_tilde @ Gamma_tilde
    comp_2_2 = comp_2_1 + E_tilde
    comp_2_3 = X_tilde @ comp_2_2
    second_term = (rd) * comp_2_3

    g_1 = first_term + second_term
    g_1 = np.array(g_1)


    # g_2
    comp_1 = (0.5 * pow(sigma, 2))
    first_term = comp_1 * Gamma_tilde @ D_vv_tilde.transpose() @ Y_tilde

    comp_2 = kappa * Gamma_tilde
    second_term = comp_2 @ D_v_tilde.transpose() @ (eta * I_v_tilde - Y_tilde)

    g_2 = first_term + second_term



    g_0_sliced = g_0[np.ix_(index_s_boundary, index_v_boundary)]
    g_0_vector = g_0_sliced.flatten(order='F')


    g_1_sliced = g_1[np.ix_(index_s_boundary, index_v_boundary)]
    g_1_vector = g_1_sliced.flatten(order='F')

    g_2_sliced = g_2[np.ix_(index_s_boundary, index_v_boundary)]
    g_2_vector = g_2_sliced.flatten(order='F')

    # Ensure all vectors have the same shape
    g_0 = g_0_vector
    g_1 = g_1_vector
    g_2 = g_2_vector


    g = g_0 + g_1 + g_2



    if scheme == 'Do':

        theta = 0.5

    if scheme == 'CS':

        theta = 0.5

    if scheme == 'MCS':

        theta = 1/3

    if scheme == 'HV':

        theta = 1/2 + ((1/6) * math.sqrt(3))

    if theta_exp != 0.0:

        theta = theta_exp

    #start3 = datetime.now()





    start = datetime.now()  # Start timing ADI loop from LU decomp

    # LU Decomposition
    I = kron(I_v, I_s)

    LHS_2 = sp.eye(I.shape[0]) - (theta * dt) * A_2

    LHS_1 = sp.eye(I.shape[0]) - (theta * dt) * A_1

    # Decompose A_2 part
    LU1 = spla.splu(LHS_1)

    # Decompose A_1 part
    LU2 = spla.splu(LHS_2)










    if scheme == 'Do':

        damping_Do = False

        if damping_Do:
            print("Damping")

            half_point = 0.5 * dt
            I = sp.identity(A.shape[0])  # Assuming A is square

            # Two Backward Euler steps involving A     --     Decompose the matrix A
            matrix_sparse = sp.eye(I.shape[0]) - half_point * A
            LU = spla.splu(matrix_sparse)

            # First solve step
            u = LU.solve(u + half_point *  g)

            # Second solve step)
            u = LU.solve(u + half_point * g)




        for n in range(1, N + 1):
            gr = g
            gr0 = g_0
            gr1 = g_1
            gr2 = g_2
            u = Do(u, dt, LU1, LU2, A, g, g_1, gr1, g_2, gr2, theta)
            g = gr
            g_0 = gr0
            g_1 = gr1
            g_2 = gr2


    if scheme == 'CS':

        damping_CS = False

        if damping_CS:
            print("Damping")

            half_point = 0.5 * dt
            I = sp.identity(A.shape[0])

            # Two Backward Euler steps involving A     --     Decompose the matrix A
            matrix_sparse = sp.eye(I.shape[0]) - half_point * A
            LU = spla.splu(matrix_sparse)

            # First solve step
            u = LU.solve(u + half_point * g)

            # Second solve step
            u = LU.solve(u + half_point * g)


            ###### Attempting Douglas as a damp, works slightly better but still looks same order as Douglas for CS.
            """
            # Predictor step
            Y0 = half_point * (A @ u + g)

            # Two corrector steps
            Y = LU1.solve(z0 + theta * half_point * (g_1 - g_1))
            Y = LU2.solve(z + theta * half_point * (g_2 - g_2))

            # Final update
            u = u + Y
            """

        for n in range(1, N + 1):
            gr = g
            gr0 = g_0
            gr1 = g_1
            gr2 = g_2
            u = CS(u, dt, LU1, LU2, A, A_0, g, g_0, g_1, g_2, gr, gr0, gr1, gr2, theta)
            g = gr
            g_0 = gr0
            g_1 = gr1
            g_2 = gr2

    if scheme == 'MCS':
        for n in range(1, N + 1):
            gr = g
            gr0 = g_0
            gr1 = g_1
            gr2 = g_2
            u = MCS(u, dt, LU1, LU2, A, A_0, g, g_0, g_1, g_2, gr, gr0, gr1, gr2, theta)
            g = gr
            g_0 = gr0
            g_1 = gr1
            g_2 = gr2

    if scheme == 'HV':
        for n in range(1, N + 1):
            gr = g
            gr0 = g_0
            gr1 = g_1
            gr2 = g_2
            u = HV(u, dt, LU1, LU2, A, A_0, g, g_0, g_1, g_2, gr, gr0, gr1, gr2, theta)
            g = gr
            g_0 = gr0
            g_1 = gr1
            g_2 = gr2

    end = datetime.now()
    time = (end - start).total_seconds()

    #print("Computation Time (ADI loop):", time)

    # Reshape for plotting + Greeks calculations   -  First is forward then central after
    temp_vec = u

    U = np.zeros((m1_tilde, m2_tilde))

    U[np.ix_(index_s_boundary, index_v_boundary)] = temp_vec.reshape((index_s_boundary_size, index_v_boundary_size), order='F')

    # Ensure boundaries for Greeks work at S_0

    D_s_tilde[0, :] = 0
    Delta_s_i_1 = Delta_s[0]
    Delta_s_i_2 = Delta_s[1]

    D_s_tilde[0, 0] = (- 2 * Delta_s_i_1 - Delta_s_i_2) / (Delta_s_i_1 * (Delta_s_i_1 + Delta_s_i_2))

    D_s_tilde[0, 1] = (Delta_s_i_1 + Delta_s_i_2) / (Delta_s_i_1 * Delta_s_i_2)

    D_s_tilde[0, 2] = -Delta_s_i_1 / (Delta_s_i_2 * (Delta_s_i_1 +Delta_s_i_2))


    D_ss_tilde[0, :] = 0

    D_ss_tilde[0, 0] = 2 / (Delta_s_i_1 * (Delta_s_i_1 +Delta_s_i_2))

    D_ss_tilde[0, 1] = - 2 / (Delta_s_i_1 * Delta_s_i_2)

    D_ss_tilde[0, 2] = 2 / (Delta_s_i_2 * (Delta_s_i_1 + Delta_s_i_2))

    D_s_tilde = D_s_tilde.tocsr()
    D_ss_tilde = D_ss_tilde.tocsr()
    D_v_tilde = D_v_tilde.tocsr()


    # the Greeks
    Delta = D_s_tilde @ U
    Gamma = D_ss_tilde @ U
    Vega = U @ D_v_tilde.transpose()

    def truncate_for_plot(arr, cutoff_left, cutoff_right):
        return arr[:cutoff_left, :cutoff_right]

    S = truncate_for_plot(S, index_left_plot, index_v_l1)
    V = truncate_for_plot(V, index_left_plot, index_v_l1)
    U = truncate_for_plot(U, index_left_plot, index_v_l1)
    Delta = truncate_for_plot(Delta, index_left_plot, index_v_l1)
    Gamma = truncate_for_plot(Gamma, index_left_plot, index_v_l1)
    Vega = truncate_for_plot(Vega, index_left_plot, index_v_l1)

    # Plots (set to True or False to plot only price surface, or all, or none)
    plotPS = False

    if plotPS:
        # Calculate maxU
        maxU = np.max(U)

        # Create the plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        surf = ax.plot_surface(S, V, U, cmap='viridis')

        # Set limits
        ax.set_xlim(0, 2 * K)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, maxU)


        # Set labels and title
        ax.set_xlabel('S', fontsize=14)
        ax.set_ylabel('V', fontsize=14)
        ax.set_zlabel('Option Value', fontsize=14)
        ax.set_title('Option value under Heston', fontsize=14)

        # Add color bar which maps values to colors
        fig.colorbar(surf)

        # Show the plot
        plt.show()



    plot_all = False

    if plot_all:
        # Plotting option value under heston

        # Calculate maxU
        maxU = np.max(U)

        # Create the plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        surf = ax.plot_surface(S, V, U, cmap='viridis')



        # Set labels and title
        ax.set_xlabel('S', fontsize=14)
        ax.set_ylabel('V', fontsize=14)
        ax.set_zlabel('Option Value', fontsize=14)
        ax.set_title('Option value under Heston', fontsize=14)

        # Add color bar which maps values to colors
        fig.colorbar(surf)

        # Show the plot
        plt.show()

        # Plot Delta
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, projection='3d')
        surf1 = ax1.plot_surface(S, V, Delta, cmap='viridis')
        ax1.set_xlim([0, 2 * K])
        ax1.set_ylim([0, 1])
        ax1.set_xlabel('S', fontsize=14)
        ax1.set_ylabel('V', fontsize=14)
        ax1.set_title('Delta', fontsize=14)
        fig1.colorbar(surf1)
        plt.show()

        # Plot Gamma
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')
        surf2 = ax2.plot_surface(S, V, Gamma, cmap='viridis')
        ax2.set_xlim([0, 2 * K])
        ax2.set_ylim([0, 1])
        ax2.set_xlabel('S', fontsize=14)
        ax2.set_ylabel('V', fontsize=14)
        ax2.set_title('Gamma', fontsize=14)
        fig2.colorbar(surf2)
        plt.show()

        # Plot Vega
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111, projection='3d')
        surf3 = ax3.plot_surface(S, V, Vega, cmap='viridis')
        ax3.set_xlim([0, 2 * K])
        ax3.set_ylim([0, 1])
        ax3.set_xlabel('S', fontsize=14)
        ax3.set_ylabel('V', fontsize=14)
        ax3.set_title('Vega', fontsize=14)
        fig3.colorbar(surf3)
        plt.show()




    #   Getting specific Option Values - First interpolator is just nearest grid point

    # Create the interpolator using NearestNDInterpolator
    points = np.array([(S[i, j], V[i, j]) for i in range(S.shape[0]) for j in range(S.shape[1])])
    values = U.flatten()
    nearest_interpolator = NearestNDInterpolator(points, values)

    # Specify the current values of S and V
    S_current = 120.0  # Replace with the current value of S
    V_current = 0.4  # Replace with the current value of V

    # Interpolate to get the option price at the current S and V
    option_price = nearest_interpolator((S_current, V_current))

    true_price = heston_price(S_current, K, T, rd, rf, V_current, kappa, eta, sigma, rho)

    #true_price = 45.63379764919681

    print("TRUE PRICE: ", true_price)

    #print("(Nearest grid point (bad)) Option price at S =", S_current, "and V =", V_current, "is", option_price)

    #print("(Nearest grid point (bad)) Relative error:", abs((option_price - true_price) / true_price))

    # Better interpolator
    points = np.array([(S[i, j], V[i, j]) for i in range(S.shape[0]) for j in range(S.shape[1])])
    values = U.flatten()

    linear_interpolator = LinearNDInterpolator(points, values)

    option_price_linear = linear_interpolator(S_current, V_current)

    print("(LinearNDInterpolator) Option price at S =", S_current, "and V =", V_current, "using LinearNDInterpolator is", option_price_linear)

    print("(LinearNDInterpolator) Relative error :", abs((option_price_linear - true_price) / true_price))

    # Return for global spatial/temporal functions
    return U, S, V


start4 = datetime.now()

FDMHeston()

end4 = datetime.now()
time4 = (end4 - start4).total_seconds()

#print("Computation time (All): ", time4)






##############################################################################################













def compute_global_spatial_error(m1_values, kappa, eta, sigma, rho, rd, rf, T, K, scheme):
    errors = []

    for m1 in m1_values:



        U, S, V = FDMHeston(kappa=kappa, eta=eta, sigma=sigma, rho=rho, rd=rd, rf=rf, T=T, K=K, m1=m1, scheme=scheme, N=500, theta_exp=0.0)
        max_error = 0

        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                if 0 < V[i, j] < 1 and (0.5 * K) < S[i, j] < (1.5 * K):
                    analytical_price = heston_price(S[i, j], K, T, rd, rf, V[i, j], kappa, eta, sigma, rho)
                    max_error = max(max_error, abs(U[i, j] - analytical_price))
                    print("MAX ERROR :", max_error, "At Point S=", S[i, j], "At Point V=", V[i, j], "Price FDM  = ", U[i,j], "Price Analytical =", analytical_price)


        errors.append(max_error)

    return errors



# Params
#case 3
kappa = 0.38
eta = 0.09
sigma = 1.26
rho = -0.55
rd = 0.01
rf = 0.0
T = 4
K = 100

#case 4
#kappa = 0.3
#eta = 0.06
#sigma = 0.15
#rho = 0.78
#rd = 0.01
#rf = 0.0
#T = 5
#K = 100

#Case 2
#kappa = 0.3
#eta = 0.06
#sigma = 0.15
#rho = 0.78
#rd = 0.01
#rf = 0.0
#T = 2
#K = 100


#m1_values = [10, 50, 100, 150, 200, 250, 300, 350]
m1_values = [10, 50, 100, 150, 200, 250]
m1_values = [10, 50, 100, 150, 200]


schemes = ['Do', 'CS', 'MCS', 'HV']
#schemes = ['CS']

errors_dict = {}


plot_spatial = False

if plot_spatial:

    for scheme in schemes:
        errors = compute_global_spatial_error(m1_values, kappa, eta, sigma, rho, rd, rf, T, K, scheme)
        errors_dict[scheme] = errors

    m2_values = [m1 / 2 for m1 in m1_values]

    # Plotting
    plt.figure(figsize=(10, 8))

    for scheme in schemes:
        plt.loglog([1 / m2 for m2 in m2_values], errors_dict[scheme], marker='o', label=f'Scheme: {scheme}')

    plt.xlabel('1/m2', fontsize=14)
    plt.ylabel('Spatial error', fontsize=14)
    plt.title('Global Spatial Error vs 1/m2', fontsize=16)
    plt.grid(True, which="both", ls="--")
    plt.legend(fontsize=12)
    plt.show()


"""
# Spatial error plotting - Single scheme

errors = compute_global_spatial_error(m1_values, kappa, eta, sigma, rho, rd, rf, T, K)
m2_values = [m1 / 2 for m1 in m1_values]

# Plotting
plt.figure(figsize=(8, 6))
plt.loglog([1/m2 for m2 in m2_values], errors, 'bo-')
plt.xlabel('1/m2', fontsize=14)
plt.ylabel('Spatial error', fontsize=14)
plt.title('Global Spatial Error vs 1/m2', fontsize=16)
plt.grid(True, which="both", ls="--")
plt.show()
"""



# Temporal error
def compute_global_temporal_error(N_values, kappa, eta, sigma, rho, rd, rf, T, K, scheme):
    errors = []

    #U_acc, S_acc, V_acc = FDMHeston(kappa=kappa, eta=eta, sigma=sigma, rho=rho, rd=rd, rf=rf, T=T, K=K, m1=200 , N=5000, scheme='Do')

    U_acc, S_acc, V_acc = FDMHeston(kappa=kappa, eta=eta, sigma=sigma, rho=rho, rd=rd, rf=rf, T=T, K=K, N=2000, scheme='MCS')

    for N in N_values:

        U, S, V = FDMHeston(kappa=kappa, eta=eta, sigma=sigma, rho=rho, rd=rd, rf=rf, T=T, K=K, N=N, scheme=scheme)
        max_error = 0





        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                if 0.6 < V[i, j] < 1.4 and (0.5 * K) < S[i, j] < (1.5 * K):

                    #acc_price = heston_price(S[i, j], K, T, rd, rf, V[i, j], kappa, eta, sigma, rho)

                    acc_price = U_acc[i, j]



                    max_error = max(max_error, abs(acc_price - U[i, j] ))
                    print("MAX ERROR :", max_error, "At Point S=", S[i, j], "At Point V=", V[i, j], "Price FDM  = ", U[i,j], "Price Analytical =", acc_price)
                    print("N: ", N)




        errors.append(max_error)

    return errors


plot_temporal = False

if plot_temporal:

    schemes = ['Do', 'CS', 'MCS']
    errors_dict = {}


    N_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70 ,100, 200, 300, 400, 500, 600, 700, 1200]

    for scheme in schemes:
        print("Scheme (First loop computing):", scheme)

        errors = compute_global_temporal_error(N_values, kappa, eta, sigma, rho, rd, rf, T, K, scheme)
        errors_dict[scheme] = errors

    oneOverN = [1 / N for N in N_values]

    # Plotting
    plt.figure(figsize=(10, 8))

    for scheme in schemes:
        print("Scheme (Second loop plotting):", scheme)

        plt.loglog(oneOverN, errors_dict[scheme], marker='o', label=f'Scheme: {scheme}')

    plt.xlabel('1/N', fontsize=14)
    plt.ylabel('Temporal error', fontsize=14)
    plt.title('Global Temporal Error vs 1/N', fontsize=16)
    plt.grid(True, which="both", ls="--")
    plt.legend(fontsize=12)
    plt.show()




#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################




