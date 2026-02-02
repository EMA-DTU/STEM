import torch
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import tqdm
"""
    Numerical experiment for stochastic mechanism design. A mechanism designer procures fixed demand D from multiple uncertain sources and a dispatchable source such that expected system cost is minimized.
"""


class StochMechDesign:
    def __init__(self, ):
        self.D = 100 # fixed demand
        self.mu = 25 # mean capacity of uncertain sources
        self.real_sigma = [1, 2, 4, 8, 16] # true std dev of uncertain sources
        self.sigma = self.real_sigma.copy() # reported std dev of uncertain sources
        self.a_1 = 1
        self.b_1 = 2
        self.a_2 = 2
        self.b_2 = 1.5
        self.utilityD = 5000 # utility of demand
        self.n = len(self.sigma) # number of uncertain sources
        self.batch_size = 100 # batch size for stochastic approximation of gradient in SGD

    def update_params(self, params: dict):
        for key, value in params.items():
            setattr(self, key, value)

    def c1(self, x1):
        return self.a_1 * np.pow(x1[-1], self.b_1)

    def c2(self, x1, x2):
        return self.a_2 * np.pow(abs(x2), self.b_2)

    def f1(self, sigma: list):
        # Here we solve a stochastic optimization problem using CVXPY and projected SGD, and return the first-stage decision.

        n = len(sigma)

        x1 = torch.tensor([0.5 for _ in range(n)] + [0], requires_grad=True, dtype=torch.float32)
        sigma = torch.tensor(sigma, dtype=torch.float32)
        optimizer = torch.optim.SGD([x1], lr=0.01)

        for _ in range(1000):
            x1_prev = x1.detach().numpy().copy()
            optimizer.zero_grad()

            zeta = torch.normal(mean=0, std=1, size=(self.batch_size, n)) # (batch_size, n)

            loss = self.a_1 * torch.pow(x1[-1], self.b_1) + self.a_2 * torch.mean(torch.pow(torch.abs(torch.mul(torch.mul(sigma, x1[:-1]), zeta).sum(dim=1)), self.b_2))

            loss.backward()
            optimizer.step()

            # projection step
            with torch.no_grad():
                qp = gp.Model("Projection QP")
                qp.setParam('OutputFlag', 0)
                xprime = qp.addVars(n+1, lb = 0, ub = [[1]*n + [self.D]])
                y = x1.numpy()
                qp.setObjective(gp.quicksum((xprime[i] - y[i]) * (xprime[i] - y[i]) for i in range(n+1)), GRB.MINIMIZE)
                qp.addConstr(gp.quicksum(self.mu * xprime[i] for i in range(n)) + xprime[n] == self.D)
                qp.optimize()
                if qp.status != GRB.OPTIMAL:
                    print("QP not optimal: ", qp.status)
                x1[:] = torch.tensor([xprime[i].x for i in range(n+1)], dtype=torch.float32)

                # stopping criterion
                if np.linalg.norm(x1.numpy() - x1_prev, 2) < 1e-5:
                    # print("Converged at epoch ", epoch)
                    break
        # print("Last error: ", np.linalg.norm(x1.detach().numpy() - x1_prev, 2))
        return x1.detach().numpy()

    def f2(self, sigma: list, x1, theta: list):
        # returns the required amount of reserve activation (negative value indicates downregulation)
        n = len(sigma)
        assert len(sigma) == len(theta), "Length of sigma and theta must be the same"
        assert len(x1) == n + 1, "Length of x1 must be n + 1"
        return -(sum([theta[i]*x1[i] for i in range(n)]) + x1[n] - self.D)

    def h1(self, sigma: list, i):
        # Computes h_i^1 (sigma_{-i})
        x1_i = self.f1(sigma[:i] + sigma[i+1:])
        return self.c1(x1_i)

    def t1(self, sigma: list, i):
        x1 = self.f1(sigma)
        return -self.c1(x1) + self.h1(sigma, i)

    def g2(self, sigma: list, theta: list, i):
        x1_i = self.f1(sigma[:i] + sigma[i+1:])
        x2_i = self.f2(sigma[:i] + sigma[i+1:], x1_i, theta[:i] + theta[i+1:])
        return self.c2(x1_i, x2_i)

    def t2(self, sigma: list, x1, theta: list, i):
        x2 = self.f2(sigma, x1, theta)
        return -self.c2(x1, x2) + self.g2(sigma, theta, i)

    def social_welfare(self, theta: list, x1, x2):
        return self.utilityD - self.c1(x1) - self.c2(x1, x2)
    
    def first_stage_outcome(self, ):
        return self.f1(self.sigma)

    def second_stage_outcome(self, x1, seed=0) -> tuple:
        np.random.seed(seed)
        theta_sample = list(np.random.normal(self.mu, self.real_sigma)) # realized production
        return self.f2(self.sigma, x1, theta_sample), theta_sample
        
    def average_outcomes(self, num_samples=100):
        x1 = self.first_stage_outcome()
        payment1 = [self.t1(self.sigma, i) for i in range(self.n)]
        sw_values = []
        budget_balances = []
        payments = []
        for j in range(num_samples):
            x2, theta_sample = self.second_stage_outcome(x1, seed=j)
            payment2 = [self.t2(self.sigma, x1, theta_sample, i) for i in range(self.n)]
            sw_values.append(self.social_welfare(theta_sample, x1, x2))
            budget_balances.append( -np.sum(payment1) - np.sum(payment2) - self.c1(x1) - self.c2(x1, x2) + self.utilityD )
            payments.append( [payment1[i] + payment2[i] for i in range(self.n)] )
        return np.mean(sw_values), np.mean(budget_balances), np.mean(payments, axis=0)

def reserve_impact():
    # impact of reserve cost parameters on first stage decision
    market = StochMechDesign()
    dispatchable_power = []
    a_2_values = [0.5, 1, 1.5, 2, 3, 4]
    for a_2 in a_2_values:
        market.update_params({'a_2': a_2})
        f1 = market.first_stage_outcome()
        dispatchable_power.append(f1[-1])
    
    fig = plt.figure()
    plt.rcParams.update({'font.size': 14})
    ax = fig.add_subplot(1,1,1)
    ax.plot(a_2_values, dispatchable_power, marker='o')
    ax.set_xlabel('Reserve Cost Parameter a_2')
    ax.set_ylabel('Dispatchable Power Procured')
    ax.set_title('Impact of Reserve Cost on Dispatchable Power Procurement')
    plt.show()
    # print(dispatchable_power)

    return None
    
def dispatchable_impact():
    # impact of reserve cost parameters on first stage decision
    market = StochMechDesign()
    dispatchable_power = []
    a_1_values = [0.5, 1, 1.5, 2, 3, 4]
    for a_1 in a_1_values:
        market.update_params({'a_1': a_1})
        f1 = market.first_stage_outcome()
        dispatchable_power.append(f1[-1])
    
    fig = plt.figure()
    plt.rcParams.update({'font.size': 14})
    ax = fig.add_subplot(1,1,1)
    ax.plot(a_1_values, dispatchable_power, marker='o')
    ax.set_xlabel('Reserve Cost Parameter a_1')
    ax.set_ylabel('Dispatchable Power Procured')
    ax.set_title('Impact of dispatchable power cost on Dispatchable Power Procurement')
    plt.show()
    # print(dispatchable_power)

    return None
    
def dynamic_vs_static():
    # so in static, participants will lie but their real sigma will remain the same
    market_dynamic = StochMechDesign()
    market_static = StochMechDesign()
    market_static.update_params({'sigma': [0]*market_static.n}) # participants report zero uncertainty in static mechanism

    dynamic_sw, dynamic_budget_balance, dynamic_payments = market_dynamic.average_outcomes()
    static_sw, static_budget_balance, static_payments = market_static.average_outcomes()

    print("Dynamic:")
    print("First-stage decision: ", market_dynamic.first_stage_outcome())
    print("Payments: ", dynamic_payments)
    print("Average social welfare: ", dynamic_sw)

    print("Static:")
    print("First-stage decision: ", market_static.first_stage_outcome())
    print("Payments: ", static_payments)
    print("Average social welfare: ", static_sw)

    return None


if __name__ == "__main__":
    np.random.seed(0)
    
    # reserve_impact()
    # dispatchable_impact()
    dynamic_vs_static()

    # outcome1 = f1(sigma)
    # print("First-stage decision: ", outcome1)
    # payment1 = [t1(sigma, i) for i in range(n)]
    # print("First-stage payments: ", payment1)

    # payment2 = []
    # budget_balance = []
    # for _ in tqdm.tqdm(range(50)):
    #     theta_sample = list(np.random.normal(mu, sigma))
    #     outcome2 = f2(sigma, outcome1, theta_sample)

    #     payment2.append([t2(sigma, outcome1, theta_sample, i) for i in range(n)])

    #     budget_balance.append(-np.sum(payment1) - np.sum(payment2[-1]) - c1(outcome1) - c2(outcome1, outcome2) + utilityD)

    #     # print("Second-stage decision (reserve activation): ", outcome2)
    #     # print("Second-stage payment: ", t2(outcome1, outcome2))

    # payment2 = np.array(payment2)

    # [print(f"Average total payment for player {i}: ", np.mean(payment2[:,i] + payment1[i])) for i in range(n)]
    # print("Average budget balance: ", np.mean(budget_balance))
