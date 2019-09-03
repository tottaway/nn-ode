import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from activations import activations
 

def forward(X, W, U, V, activation, activation_p):
    Z = W @ X + U
    A = activation(Z)
    N = V @ A
 
    new_A = activation_p(Z)
    # calculate P_i*V_i in this case P == W
    new_V = np.multiply(W.T, V)
    # print("new_A: " + str(new_A.shape))
    # print("new_V: " + str(new_V.shape))
    N_g = new_V @ new_A
    return N, N_g, A, Z


def compute_cost(y_0, X_i, N, N_g):
    y = y_0 + X_i * N
    dy_dx = N + (X_i * N_g)
    sqrt_cost = dy_dx - f(X_i, y)
    cost = sqrt_cost ** 2
    return cost, sqrt_cost
 

def backwards(activation_p, activation_pp, sqrt_cost, Z, A, X_i, V, W):
    # gp = g_prime
    sigma_p = activation_p(Z)
    # gpp == g_double_prime
    sigma_pp = activation_pp(Z)

    P = W.T

    dV = 2 * sqrt_cost * (A.T + (X_i * np.multiply(P, sigma_p.T)))
    dW = (
        2 * sqrt_cost *
        (
            np.multiply(V.T, sigma_p) * X_i +
            X_i * V.T * P.T * sigma_pp + V.T * sigma_p
        )
    )
    dU = 2 * sqrt_cost * (
        np.multiply(V, P).T * sigma_pp
    )

    return dV, dW, dU


def init_values(interval, dx, n_h):
    Nx = int((interval[1] - interval[0]) / dx)
    X = np.linspace(interval[0], interval[1], Nx)
    W = np.random.randn(n_h, 1)
    U = np.random.randn(n_h, 1)
    V = np.random.randn(1, n_h)
 
    return X, W, U, V
 

def f(X, Y):
    return np.cos(X)
 

def exact_solution(X):
    return np.sin(X)
 

def display_results(X, W, U, V, activation, activation_p, y_0):
    exact = exact_solution(X)
    exact_der = f(X, exact)
    my_solution = []
    my_der = []
    for x in X:
        x_matrix = np.array([[x]])
        ans_matrix, der_matrix = forward(x_matrix, W, U, V, activation, activation_p)[0:2]
        ans = y_0 + x*ans_matrix
        der = ans_matrix + x * der_matrix
        my_solution.append(ans[0][0])
        my_der.append(der[0][0])
    my_solution = np.array(my_solution)
    my_der = np.array(my_der)
 
 
 
    fig1 = plt.subplot("121")
    plt.plot(X, exact)
    plt.plot(X, my_solution)
    fig2 = plt.subplot("122")
    plt.plot(X, exact_der)
    plt.plot(X, my_der)
    plt.show()
 
 
def main():
    interval = (0, 20)
    dx = 0.2
    n_h = 15
    y_0 = 0

    X, W, U, V = init_values(interval, dx, n_h)
    m = len(X)

    # activation funciton and derivatives
    activation, activation_p, activation_pp  = activations["sine"]

    epochs = 4000
    alpha = 0.00002

    for epoch in tqdm(range(epochs)):
        dV = 0
        dW = 0
        dU = 0

        for i in range(m):
            X_i = np.array([[X[i]]])
            N, N_g, A, Z = forward(X_i, W, U, V, activation, activation_p)
            cost, sqrt_cost = compute_cost(y_0, X_i, N, N_g)
            dV_i, dW_i, dU_i = backwards(activation_p, activation_pp, sqrt_cost, Z, A, X_i, V, W)
            dV += dV_i
            dU += dU_i
            dW += dW_i

        V -= dV * alpha
        W -= dW * alpha
        U -= dU * alpha
        if epoch % 800 == 0:
            print(cost)
            display_results(X, W, U, V, activation, activation_p, y_0)

    display_results(X, W, U, V, activation, activation_p, y_0)


if __name__ == "__main__":
    main()

