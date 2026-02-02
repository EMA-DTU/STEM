import torch
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import tqdm
"""
    Numerical experiment for stochastic mechanism design. A mechanism designer procures fixed demand D from multiple uncertain sources and a dispatchable source such that expected system cost is minimized.
"""

# parameters
D = 100
mu = 20
sigma = [1, 2, 4, 8, 16]
a_1 = 1
b_1 = 2
a_2 = 2
b_2 = 1.5
utilityD = 5000
n = len(sigma)
batch_size = 100

def c1(x1):
    return a_1 * np.pow(x1[-1], b_1)

def c2(x1, x2):
    return a_2 * np.pow(abs(x2), b_2)

# @profile
def f1(sigma, ):
    # Here we solve a stochastic optimization problem using CVXPY and projected SGD, and return the first-stage decision.
    n = len(sigma)

    x1 = torch.tensor([0.5 for _ in range(n)] + [0], requires_grad=True, dtype=torch.float32)
    sigma = torch.tensor(sigma, dtype=torch.float32)
    optimizer = torch.optim.SGD([x1], lr=0.01)

    for epoch in range(1000):
        x1_prev = x1.detach().numpy().copy()
        optimizer.zero_grad()

        zeta = torch.normal(mean=0, std=1, size=(batch_size, n)) # (batch_size, n)

        loss = a_1 * torch.pow(x1[-1], b_1) + a_2 * torch.mean(torch.pow(torch.abs(torch.mul(torch.mul(sigma, x1[:-1]), zeta).sum(dim=1)), b_2))

        loss.backward()
        optimizer.step()

        # projection step
        with torch.no_grad():
            qp = gp.Model("Projection QP")
            qp.setParam('OutputFlag', 0)
            xprime = qp.addVars(n+1, lb = 0, ub = [[1]*n + [D]])
            y = x1.numpy()
            qp.setObjective(gp.quicksum((xprime[i] - y[i]) * (xprime[i] - y[i]) for i in range(n+1)), GRB.MINIMIZE)
            qp.addConstr(gp.quicksum(mu * xprime[i] for i in range(n)) + xprime[n] == D)
            qp.optimize()
            if qp.status != GRB.OPTIMAL:
                print("QP not optimal: ", qp.status)
            x1[:] = torch.tensor([xprime[i].x for i in range(n+1)], dtype=torch.float32)



            # xprime = cp.Variable(n+1, nonneg=True)
            
            # P = np.eye(n+1)
            # A = np.array([[mu]*n + [1]])
            # b = np.array([D])
            # G = np.eye(n+1)
            # h = np.concat((np.ones((n,1)), np.array([[D]])), 0)
            # prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(xprime - x1.numpy(), P)),
            #                   [A @ xprime == b,
            #                    G @ xprime <= h])
            # prob.solve()
            # if prob.status not in ['optimal']:
            #     print(prob.status)
            #     print(A)
            #     print(b)
            #     print(G)
            #     print(h)

            # P = np.eye(n+1)
            # obj = cp.Minimize(0.5 * cp.quad_form(xprime - x1.numpy(), P))
            # cons = [cp.sum(xprime[i] * mu for i in range(n)) + xprime[n] == D] + [xprime[i] <= 1 for i in range(n)] + [xprime[n] <= D]
            # problem = cp.Problem(obj, cons)
            # problem.solve()

            # x1[:] = torch.tensor(xprime.value, dtype=torch.float32)

            # stopping criterion
            if np.linalg.norm(x1.numpy() - x1_prev, 2) < 1e-5:
                print("Converged at epoch ", epoch)
                break
    print("Last error: ", np.linalg.norm(x1.detach().numpy() - x1_prev, 2))
    return x1.detach().numpy()

def f2(sigma, x1, theta):
    # returns the required amount of reserve activation (negative value indicates downregulation)
    n = len(sigma)
    assert len(sigma) == len(theta), "Length of sigma and theta must be the same"
    assert len(x1) == n + 1, "Length of x1 must be n + 1"
    return -(sum([theta[i]*x1[i] for i in range(n)]) + x1[n] - D)

def t1(sigma, i):
    x1 = f1(sigma)
    x1_i = f1(sigma[:i] + sigma[i+1:])
    return -c1(x1) + c1(x1_i)

def t2(sigma, x1, theta, i):
    x1_i = f1(sigma[:i] + sigma[i+1:])
    x2_i = f2(sigma[:i] + sigma[i+1:], x1_i, theta[:i] + theta[i+1:])
    x2 = f2(sigma, x1, theta)
    return -c2(x1, x2) + c2(x1_i, x2_i)


if __name__ == "__main__":
    outcome1 = f1(sigma)
    print("First-stage decision: ", outcome1)
    payment1 = [t1(sigma, i) for i in range(n)]
    print("First-stage payments: ", payment1)

    payment2 = []
    budget_balance = []
    for _ in tqdm.tqdm(range(50)):
        theta_sample = list(np.random.normal(mu, sigma))
        outcome2 = f2(sigma, outcome1, theta_sample)

        payment2.append([t2(sigma, outcome1, theta_sample, i) for i in range(n)])

        budget_balance.append(-np.sum(payment1) - np.sum(payment2[-1]) - c1(outcome1) - c2(outcome1, outcome2) + utilityD)

        # print("Second-stage decision (reserve activation): ", outcome2)
        # print("Second-stage payment: ", t2(outcome1, outcome2))

    payment2 = np.array(payment2)
    fig, axs = plt.subplots(1, 2)
    axs[0].hist(payment2[:,0] + payment1[0], 20)
    axs[0].set_title("Histogram of total payments for player 1")
    axs[0].set_xlabel("Payment Amount")

    axs[1].hist(budget_balance, 20)
    axs[1].set_title("Histogram of Budget Balance")
    axs[1].set_xlabel("Budget Balance Amount")
    plt.show()
