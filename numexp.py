import torch
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from line_profiler import profile
import warnings
import copy
torch.manual_seed(0)
"""
    Numerical experiment for stochastic mechanism design. A mechanism designer procures fixed demand D from multiple uncertain sources and a dispatchable source such that expected system cost is minimized.
"""


class StochasticMarket:
    def __init__(self, delta: list = [[[-5, 20, 100], [0, 1, 0]],
                                      [[-4, 20, 100], [0, 2, 0]],
                                      [[-3, 20, 100], [0, 4, 0]],
                                      [[-2, 20, 100], [0, 6, 0]],
                                      [[-1, 20, 100], [0, 8, 0]],
                                    #   [[-100, 20, 100], [0, 16, 0]],
                                    #   [[-0.5, 20, 100], [0, 6, 0]],
                                     ], 
                D: int = 100, alpha_1=3, alpha_2=7, alpha_3=8, alpha_4=50):
        self.D = D # fixed demand
        self.real_delta = delta # true cost parameters
        self.delta = copy.deepcopy(self.real_delta) # reported parameters (n x 2 x 3)
        self.alpha_1 = alpha_1 # marginal cost of reserve capacity
        self.alpha_2 = alpha_2 # marginal cost of dispatchable power
        self.alpha_3 = alpha_3 # reserve activation cost
        self.alpha_4 = alpha_4 # load shedding cost
        self.n = len(self.delta) # number of sources
        self.batch_size = 1000 # batch size for finite sample approximation of second-stage costs in first-stage optimization

        # stored first-stage decisions
        self.x1_cache = {} # indexed by excluded participant
        self.x2_scenarios_cache = None
        self.x2_cache = {} # indexed by excluded participant

        self.real_theta = None
        self.theta = None

    def update_params(self, params: dict):
        for key, value in params.items():
            setattr(self, key, value)
        self.x1_cache = {} # reset cache
        self.x2_scenarios_cache = None
        self.x2_cache = {} # reset cache

    def c1(self, x1):
        # First-stage decision cost
        return self.alpha_1 * x1[-2] + self.alpha_2 * x1[-1]

    def c2(self, x1, x2):
        # Second-stage decision cost
        return self.alpha_3 * x2[-2] + self.alpha_4 * x2[-1]

    def ci(self, thetai: list, x1, x2, i):
        # Sanity check: costs computed by gurobi should match this function
        if x2[i] <= thetai[1]:
            return thetai[0]*(x2[i] - thetai[1])
        else:
            return thetai[2]*(x2[i] - thetai[1])

    def generate_theta_samples(self, delta, n_samples, seed=0):
        theta_samples = [] # (batch_size, n, 3)
        for k in range(n_samples):
            theta_samples.append([])
            for i in range(len(delta)):
                np.random.seed(seed + k)
                theta_1 = np.random.normal(delta[i][0][0], delta[i][1][0])
                theta_2 = np.clip(np.random.normal(delta[i][0][1], delta[i][1][1]), 0, 40) # max and min capacity of producers
                theta_3 = np.random.normal(delta[i][0][2], delta[i][1][2])
                theta_samples[-1].append([theta_1, theta_2, theta_3])
        return theta_samples
    
    def x1star(self, excl: int = -1):
        # Here, we solve the first-stage market problem using Gurobipy by a finite sample approximation of the second-stage costs, and return the optimal first-stage decision.
        if excl in self.x1_cache.keys():
            return self.x1_cache[excl]

        n = self.n

        theta_samples = self.generate_theta_samples(self.delta, self.batch_size)
        
        model = gp.Model("First-stage optimization")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', 200)
        
        x1n = model.addVars(n, lb=0, ub=1, vtype=GRB.BINARY, name="x1n") # unit commitment decision for uncertain sources
        x1n1 = model.addVar(lb=0, ub=self.D, name="x1n1") # reserve capacity
        x1n2 = model.addVar(lb=0, ub=self.D, name="x1n2") # dispatchable power
        x2n = [model.addVars(n, lb=0, ub=self.D, name="x2_{}".format(k)) for k in range(self.batch_size)] # final dispatch for uncertain sources (batch_size, n)
        x2n1 = [model.addVar(lb=0, name="x2n1_{}".format(k)) for k in range(self.batch_size)] # reserve activation (batch_size)
        x2n2 = [model.addVar(lb=0, name="x2n2_{}".format(k)) for k in range(self.batch_size)] # load shedding (batch_size)
        # y2 = model.addVars(self.batch_size, lb=0, name='y2') # auxiliary variable representing abs(x2n1)
        w = [model.addVars(n, lb=0, name="w_{}".format(k)) for k in range(self.batch_size)] # auxiliary variables for linearization of second-stage dispatch constraints
        c = [model.addVars(n, lb=0, name="c_{}".format(k)) for k in range(self.batch_size)] # cost variables for producers

        model.setObjective( self.alpha_1 * x1n1 + self.alpha_2 * x1n2 + (1/self.batch_size) * ( gp.quicksum(self.alpha_3 * x2n1[k] + self.alpha_4 * x2n2[k] + gp.quicksum(c[k][i] for i in range(n)) for k in range(self.batch_size)) ), GRB.MINIMIZE)

        [model.addGenConstrPWL(x2n[k][i], c[k][i], [0, theta_samples[k][i][1], self.D], [-theta_samples[k][i][0]*theta_samples[k][i][1], 0, theta_samples[k][i][2]*(self.D - theta_samples[k][i][1])]) for k in range(self.batch_size) for i in range(n)]
        
        # model.addConstrs( y2[k] >= x2n1[k] for k in range(self.batch_size) )
        # model.addConstrs( y2[k] >= -x2n1[k] for k in range(self.batch_size) )
        model.addConstrs( x2n1[k] <= x1n1 for k in range(self.batch_size) )

        model.addConstrs( (self.D - gp.quicksum(w[k][i] for i in range(n)) - x1n2 - x2n1[k] - x2n2[k] == 0 for k in range(self.batch_size)), name="balance" )
        model.addConstrs( w[k][i] <= x1n[i] * self.D for k in range(self.batch_size) for i in range(n) )
        model.addConstrs( w[k][i] <= x2n[k][i] for k in range(self.batch_size) for i in range(n) )
        model.addConstrs( w[k][i] >= x2n[k][i] - self.D * (1-x1n[i]) for k in range(self.batch_size) for i in range(n) )

        if excl != -1:
            model.addConstr( x1n[excl] == 0 )

        model.optimize()

        if model.status != GRB.OPTIMAL:
            print("Gurobi not optimal: ", model.status)
        
        x1_opt = [x1n[i].x for i in range(n)] + [x1n1.x] + [x1n2.x]
        x2_opt = [[x2n[k][i].x for i in range(n)] + [x2n1[k].x] + [x2n2[k].x] for k in range(self.batch_size)]
        self.x1_cache[excl] = x1_opt
        if excl == -1:
            self.x2_scenarios_cache = x2_opt
        return x1_opt
    
    def realize_theta(self, seed=0):
        self.real_theta = self.generate_theta_samples(self.real_delta, 1, seed=seed)[0]
        self.theta = copy.deepcopy(self.real_theta)
        self.x2_cache = {}

        return self.theta

    def x2star(self, excl: int = -1):
        # Here, we solve the second-stage market problem using Gurobipy, and return the optimal second-stage decision.
        if excl in self.x2_cache.keys():
            return self.x2_cache[excl]
        
        x1 = self.x1star(excl=excl)
        n = self.n

        model = gp.Model("Second-stage optimization")
        model.setParam('OutputFlag', 0)
        x2n = model.addVars(n, lb=0, ub=self.D, name="x2") # final dispatch for uncertain sources
        x2n1 = model.addVar(lb=0, name="x2n1") # reserve activation
        x2n2 = model.addVar(lb=0, name="x2n2") # load shedding
        # y2 = model.addVar(lb=0, name='y2') # auxiliary variable representing abs(x2n1)
        # z2 = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='z2')
        c = model.addVars(n, lb=0, name="c") # cost variables for producers
        
        model.setObjective( self.alpha_3 * x2n1 + self.alpha_4 * x2n2 + gp.quicksum( c[i] for i in range(n) ), GRB.MINIMIZE )
        
        [model.addGenConstrPWL(x2n[i], c[i], [0, self.theta[i][1], self.D], [-self.theta[i][0]*self.theta[i][1], 0, self.theta[i][2]*(self.D - self.theta[i][1])]) for i in range(n)]
        model.addConstr( self.D - gp.quicksum(x2n[i] * x1[i] for i in range(n)) - x1[-1] - x2n1 - x2n2 == 0, name="balance" )
        model.addConstr( x2n1 <= x1[-2] )

        model.optimize()

        if model.status != GRB.OPTIMAL:
            print("Gurobi not optimal: ", model.status)
        x2_opt = [x2n[i].x for i in range(n)] + [x2n1.x] + [x2n2.x]
        self.x2_cache[excl] = x2_opt
        return x2_opt

    def h1(self, i):
        # Computes h_i^1 (delta_{-i})
        x1_i = self.x1star(excl=i)
        return self.c1(x1_i)
        # return 0

    def t1(self, i):
        x1 = self.x1star()
        return -self.c1(x1) + self.h1(i)

    def g2(self, i):
        x1_i = self.x1star(excl=i)
        x2_i = self.x2star(excl=i)

        return self.c2(x1_i, x2_i) + np.sum([self.ci(self.theta[j], x1_i, x2_i, j) for j in range(self.n) if j != i])
        # return 0

    def t2(self, i):
        x1 = self.x1star()
        x2 = self.x2star()
        return self.ci(self.theta[i], x1, x2, i) - ( self.c2(x1, x2) + np.sum([self.ci(self.theta[j], x1, x2, j) for j in range(self.n)]) ) + self.g2(i)
    
    def first_stage_outcome(self, ):
        return self.x1star()

    def second_stage_outcome(self, ) -> list:
        if self.theta is None:
            warnings.warn("Theta not realized yet")
        return self.x2star()

    def system_cost(self, x1, x2, theta):
        return self.c1(x1) + self.c2(x1, x2) + np.sum([self.ci(theta[i], x1, x2, i) for i in range(self.n)])

    def average_outcomes(self, ):
        # Plots and saves all the metrics for the mechanism with variance for second-stage metrics
        metrics = {}
        x1 = self.first_stage_outcome()
        payment1 = [self.t1(i) for i in range(self.n)]
        metrics['first_stage_dispatch'] = x1[:-2]
        metrics['reserve_capacity'] = x1[-2]
        metrics['dispatchable_power'] = x1[-1]

        producer_costs = []
        payment2s = []
        payments = []
        utilities = []
        reserve_activations = []
        dispatch2s = []
        load_sheds = []
        system_costs = []
        for j in range(self.batch_size):
            theta_sample = self.realize_theta(seed=j)
            x2 = self.second_stage_outcome()
            payment2s.append([self.t2(i) for i in range(self.n)])
            payments.append( [payment1[i] + payment2s[j][i] for i in range(self.n)] )
            producer_costs.append( [self.ci(theta_sample[i], x1, x2, i) for i in range(self.n)] )
            utilities.append( [payments[j][i] - producer_costs[j][i] for i in range(self.n)] )
            reserve_activations.append(x2[-2])
            load_sheds.append(x2[-1])
            dispatch2s.append(x2[:-2])
            system_costs.append(self.system_cost(x1, x2, theta_sample))
        
        fig1 = plot_average_deviation(np.arange(1, self.n+1), producer_costs, xlabel='Producer', ylabel='Cost', title='Average Producer Costs with Std Dev')
        fig2 = plot_average_deviation([str(i+1) + 'x' if x1[i] <= 0 else str(i+1) for i in range(self.n)], [payment1, payment2s, payments], xlabel='Producer', ylabel='Payment', title='Average Payments with Std Dev', legends=['First-stage payment', 'Second-stage payment', 'Total payment'])
        fig3 = plot_average_deviation(np.arange(1, self.n+1), utilities, xlabel='Producer', ylabel='Utility', title='Average Utilities with Std Dev')
        fig4 = plot_average_deviation([1], reserve_activations, xlabel='', ylabel='Reserve Activation', title=f'Reserve Activations (Procured: {metrics["reserve_capacity"]:.2f}, Dispatchable: {metrics["dispatchable_power"]:.2f})')
        fig5 = plot_average_deviation([1], load_sheds, xlabel='', ylabel='Load Shedding', title='Load Shedding')
        fig6 = plot_average_deviation([1], system_costs, xlabel='', ylabel='System Cost', title='System Costs')

        fig1.savefig('producer_costs.pdf')
        fig2.savefig('payments.pdf')
        fig3.savefig('utilities.pdf')
        fig4.savefig('reserve_activations.pdf')
        fig5.savefig('load_sheds.pdf')
        fig6.savefig('system_costs.pdf')

        return None

def plot_average_deviation(x: list, y: list, xlabel: str, ylabel: str, title: str, legends: list = None):
    # y: (batch_size, n) or (batch_size,)
    fig = plt.figure()
    plt.rcParams.update({'font.size': 12})
    ax = fig.add_subplot(1,1,1)

    if legends is not None:
        assert len(legends) == len(y), "Length of legends must match length of y"
        for j in range(len(legends)):
            if len(np.shape(y[j])) == 2:
                y_mean = np.mean(y[j], axis=0)
                y_std = np.std(y[j], axis=0)
            else:
                y_mean = y[j]
                y_std = np.zeros_like(y_mean)
        
            ax.plot(x, y_mean, marker='o', label=legends[j])
            ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)
            ax.set_xlabel(xlabel)
            ax.legend()
            ax.set_ylabel(ylabel)
            ax.set_title(title)
    else:
        if len(np.shape(y)) == 2:
            y_mean = np.mean(y, axis=0)
            y_std = np.std(y, axis=0)
        else:
            y_mean = y
            y_std = np.zeros_like(y_mean)


    
        ax.boxplot(np.array(y), positions=x, widths=0.5)
        # ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    return fig

def dispatchable_impact():
    # impact of dispatchable cost on dispatchable power
    dispatchable = []
    market = StochasticMarket()
    alpha_2_values = [2, 3, 4, 6, 9, 11, 13]
    for alpha_2 in tqdm(alpha_2_values):
        market.update_params({'alpha_2': alpha_2})
        x1 = market.first_stage_outcome()
        dispatchable.append(x1[-1])
    
    fig = plt.figure()
    plt.rcParams.update({'font.size': 14})
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_2_values, dispatchable, marker='o')
    ax.set_xlabel('Dispatchable Cost Parameter alpha_2')
    ax.set_ylabel('Dispatchable power procured')
    ax.set_title('Impact of Dispatchable Cost on Dispatchable Power Procurement')
    plt.show()

    return None
    
def reserve_impact_on_dispatchable():
    # impact of reserve activation cost on dispatchable power
    dispatchable = []
    market = StochasticMarket()
    alpha_3_values = [5, 7, 8, 10, 12, 16]
    for alpha_3 in tqdm(alpha_3_values):
        market.update_params({'alpha_3': alpha_3})
        x1 = market.first_stage_outcome()
        dispatchable.append(x1[-1])
    
    fig = plt.figure()
    plt.rcParams.update({'font.size': 14})
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_3_values, dispatchable, marker='o')
    ax.set_xlabel('Reserve Activation Cost Parameter alpha_3')
    ax.set_ylabel('Dispatchable power procured')
    ax.set_title('Impact of Reserve Cost on Dispatchable Power Procurement')
    plt.show()

    return None
    
def utility_on_lying(i):
    market = StochasticMarket()
    utility = []
    delta = copy.deepcopy(market.real_delta)
    sigma_values = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16]
    for s in tqdm(sigma_values):
        delta[i][1][1] = s
        market.update_params({'delta': delta})
        utilities, reserve, final_dispatch, costs, system_costs = market.average_outcomes()
        utility.append( np.mean(utilities, axis=0)[i] )
    fig = plt.figure()
    plt.rcParams.update({'font.size': 14})
    ax = fig.add_subplot(1,1,1)
    ax.plot(sigma_values, utility, marker='o')
    ax.vlines(market.real_delta[i][1][1], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='r', linestyle='--', label='True sigma')
    ax.set_xlabel(f'Reported sigma by participant {i+1}')
    ax.set_ylabel('Average Utility Received')
    ax.set_title(f'Impact of Lying on Participant {i+1}\'s Utility')
    plt.show()
    return None

def dynamic_vs_static():
    # so in static, participants will lie but their real sigma will remain the same
    market_dynamic = StochasticMarket()
    market_static = StochasticMarket()
    delta = copy.deepcopy(market_static.real_delta)
    for i in range(market_static.n):
        delta[i][1][1] = 0
    market_static.update_params({'delta': delta}) # participants report zero uncertainty in static mechanism

    dynamic_utilities, _, _, _, dynamic_system_costs = market_dynamic.average_outcomes()
    static_utilites, _, _, _, static_system_costs = market_static.average_outcomes()

    print("Dynamic:")
    print("First-stage decision: ", market_dynamic.first_stage_outcome())
    print("Utilities: ", np.mean(dynamic_utilities, axis=0))
    print("Average system cost: ", np.mean(dynamic_system_costs, axis=0))

    print("Static:")
    print("First-stage decision: ", market_static.first_stage_outcome())
    print("Utilities: ", np.mean(static_utilites, axis=0))
    print("Average system cost: ", np.mean(static_system_costs, axis=0))

    return None


if __name__ == "__main__":
    np.random.seed(0)
    
    # reserve_impact_on_dispatchable()
    # dispatchable_impact()
    # dynamic_vs_static()
    # utility_on_lying(i=1)

    market = StochasticMarket()
    market.average_outcomes()
    # x1 = market.x1star()
    # print("First-stage dispatch: ", x1[:-2])
    # print("Dispatchable power: ", x1[-1])
    # print("Reserve capacity: ", x1[-2])
    # print("")

    # payments1 = np.round([market.t1(i) for i in range(market.n)], 2)
    # market.realize_theta()
    # print("Realized generation: ", np.round([market.theta[i][1] for i in range(market.n)], 2))
    # x2 = market.x2star()
    # payments2 = np.round([market.t2(i) for i in range(market.n)], 2)

    # print("Second-stage dispatch: ", np.round(np.multiply(x2[:-2], x1[:-2]), 2))
    # print("Second-stage reserve activation: ", np.round(x2[-2], 2))
    # print("Second-stage load shedding: ", np.round(x2[-1], 2))
    # print("")

    # print("First-stage payments: ", payments1)
    # print("Second-stage payments: ", payments2)
    # print("Total payments: ", np.round([payments1[i] + payments2[i] for i in range(market.n)], 2))

    # print("System cost: ", np.round(market.system_cost(x1, x2, market.theta), 2))



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
