import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from activations import activations


def forward(X, W, U, V, activation, activation_p, Nx):
    Z = torch.bmm(W.repeat(Nx, 1, 1), X) + U.repeat(Nx, 1, 1)
    A = activation(Z)
    N = torch.bmm(V.repeat(Nx, 1, 1), A)
 
    new_A = activation_p(Z)
    # calculate P_i*V_i in this case P == W
    new_V = torch.mul(W.t(), V)
    # print("new_A: " + str(new_A.shape))
    # print("new_V: " + str(new_V.shape))
    N_g = torch.bmm(new_V.repeat(Nx, 1, 1), new_A)
    # print("Z " + str(Z.shape))
    # print("A " + str(A.shape))
    # print("W " + str(W.shape))
    # print("U " + str(U.shape))
    # print("V " + str(V.shape))
    # print("X " + str(X.shape))
    # print("W repeated " + str(W.repeat(Nx, 1, 1).shape))
    # print("U repeated " + str(U.repeat(Nx, 1, 1).shape))
    return N, N_g, A, Z


def compute_cost(y_0, X, N, N_g):
    y = y_0 + X * N
    dy_dx = N + (X * N_g)
    sqrt_cost = dy_dx - f(X, y)
    cost = sqrt_cost ** 2
    return cost, sqrt_cost
 

def backwards(activation_p, activation_pp, sqrt_cost, Z, A, X, V, W, Nx, n_h):
    sigma_p = activation_p(Z)
    sigma_pp = activation_pp(Z)

    P = W.t()

    dV = 2 * sqrt_cost * (A.reshape(Nx, 1, n_h) + (X * (P * sigma_p.reshape(Nx, 1, n_h))))
    dW = (
        2 * sqrt_cost *
        (
            np.multiply(V.t(), sigma_p) * X +
            X * V.t() * P.t() * sigma_pp + V.t() * sigma_p
        )
    )
    dU = 2 * sqrt_cost * (
        np.multiply(V, P).t() * sigma_pp
    )

    return dV, dW, dU


def init_values(interval, dx, n_h):
    Nx = int((interval[1] - interval[0]) / dx)
    X = torch.linspace(interval[0], interval[1], Nx).reshape(Nx, 1, 1)
    W = torch.randn(n_h, 1)
    U = torch.randn(n_h, 1)
    V = torch.randn(1, n_h)
 
    return Nx, X, W, U, V
 

def f(X, Y):
    return torch.cos(X)
 

def exact_solution(X):
    return torch.sin(X)
 

def display_results(X, W, U, V, activation, activation_p, y_0, Nx):
    exact = exact_solution(X).squeeze().numpy()
    exact_der = f(X, exact).squeeze().numpy()
    my_solution = []
    my_der = []

    ans_matrix, der_matrix = forward(X, W, U, V, activation, activation_p, Nx)[0:2]

    ans = y_0 + X*ans_matrix
    der = ans_matrix + X * der_matrix

    my_solution = ans.squeeze().numpy()
    my_der = der.squeeze().numpy()
    X = X.squeeze().numpy()
 
    fig1 = plt.subplot("121")
    plt.plot(X, exact)
    plt.plot(X, my_solution)
    fig2 = plt.subplot("122")
    plt.plot(X, exact_der)
    plt.plot(X, my_der)
    plt.show()
 
 
def main():
    interval = (0, 20)
    dx = 0.5
    n_h = 50
    y_0 = 0

    m, X, W, U, V = init_values(interval, dx, n_h)

    # activation funciton and derivatives
    activation, activation_p, activation_pp  = activations["sine"]

    epochs = 3000
    alpha = 0.00001

    # costs = np.zeros(epochs)

    for epoch in tqdm(range(epochs)):
        N, N_g, A, Z = forward(X, W, U, V, activation, activation_p, m)
        cost, sqrt_cost = compute_cost(y_0, X, N, N_g)
        dV, dW, dU = backwards(activation_p, activation_pp, sqrt_cost, Z, A, X, V, W, m, n_h)

        W -= alpha * torch.sum(dW, 0)
        U -= alpha * torch.sum(dU, 0)
        V -= alpha * torch.sum(dV, 0)

        # costs[epoch] = float(torch.sum(cost).data)
        if epoch == 1000:
            alpha *= 4
        # if epoch % 1000 == 0 :
        #     display_results(X, W, U, V, activation, activation_p, y_0, m)

    # X_axis = np.linspace(1, epochs, epochs)
    # plt.plot(X_axis[100:], costs[100:])
    # plt.show()

    display_results(X, W, U, V, activation, activation_p, y_0, m)


if __name__ == "__main__":
    main()

