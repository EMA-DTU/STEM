import torch
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from line_profiler import profile
import warnings
import copy
import time
import json
torch.manual_seed(0)


class StochasticMarket:
    def __init__(self, delta: list = [
                                      [[-100, 20, 300], [0, 2, 0]],
                                      [[-100, 20, 300], [0, 4, 0]],
                                      [[-100, 20, 300], [0, 6, 0]],
                                      [[-100, 20, 300], [0, 8, 0]],
                                      [[-100, 20, 300], [0, 32, 0]],
                                    #   [[-10, 20, 100], [0, 10, 0]],
                                    #   [[-15, 20, 100], [0, 10, 0]],
                                    #   [[-20, 20, 100], [0, 10, 0]],
                                    #   [[-5, 20, 100], [0, 10, 0]],
                                    #   [[-2, 20, 100], [0, 10, 0]],
                                     ], 
                D: int = 100, alpha_1=10, alpha_2=6, alpha_3=8, alpha_4=200):
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
        return self.alpha_3 * abs(x2[-2]) + self.alpha_4 * x2[-1]

    def ci(self, thetai: list, x1, x2, i):
        # Sanity check: costs computed by gurobi should match this function
        if x2[i] <= thetai[1]:
            return thetai[0]*(x2[i] - thetai[1])
        else:
            return thetai[2]*(x2[i] - thetai[1])

    def generate_theta_samples(self, delta, n_samples, seed=0):
        np.random.seed(seed)
        theta_samples = [] # (batch_size, n, 3)
        for k in range(n_samples):
            theta_samples.append([])
            for i in range(len(delta)):
                # np.random.seed(seed + i + k)
                theta_1 = np.random.normal(delta[i][0][0], delta[i][1][0])
                theta_2 = np.clip(np.random.normal(delta[i][0][1], delta[i][1][1]), 5, 35) # max and min capacity of producers
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
        x2n = [model.addVars(n, lb = 0, ub=self.D, name="x2_{}".format(k)) for k in range(self.batch_size)] # final dispatch for uncertain sources (batch_size, n)
        x2n1 = [model.addVar(lb = -GRB.INFINITY, name="x2n1_{}".format(k)) for k in range(self.batch_size)] # reserve activation (batch_size)
        y2 = [model.addVar(lb=0, name="y2_{}".format(k)) for k in range(self.batch_size)] # auxiliary variable representing abs(x2n1) (batch_size)
        x2n2 = [model.addVar(lb=0, name="x2n2_{}".format(k)) for k in range(self.batch_size)] # load shedding (batch_size)
        # y2 = model.addVars(self.batch_size, lb=0, name='y2') # auxiliary variable representing abs(x2n1)
        w = [model.addVars(n, lb=0, name="w_{}".format(k)) for k in range(self.batch_size)] # auxiliary variables for linearization of second-stage dispatch constraints
        c = [model.addVars(n, lb=0, name="c_{}".format(k)) for k in range(self.batch_size)] # cost variables for producers

        model.setObjective( self.alpha_1 * x1n1 + self.alpha_2 * x1n2 + (1/self.batch_size) * ( gp.quicksum(self.alpha_3 * y2[k] + self.alpha_4 * x2n2[k] + gp.quicksum(c[k][i] for i in range(n)) for k in range(self.batch_size)) ), GRB.MINIMIZE)

        [model.addGenConstrPWL(x2n[k][i], c[k][i], [0, theta_samples[k][i][1], self.D], [-theta_samples[k][i][0]*theta_samples[k][i][1], 0, theta_samples[k][i][2]*(self.D - theta_samples[k][i][1])]) for k in range(self.batch_size) for i in range(n)]
        
        model.addConstrs( y2[k] >= x2n1[k] for k in range(self.batch_size) )
        model.addConstrs( y2[k] >= -x2n1[k] for k in range(self.batch_size) )
        model.addConstrs( y2[k] <= x1n1 for k in range(self.batch_size) )
        model.addConstrs( x2n[k][i] - theta_samples[k][i][1] <= 15 for i in range(n) for k in range(self.batch_size) )
        model.addConstrs( theta_samples[k][i][1] - x2n[k][i] <= 15 for i in range(n) for k in range(self.batch_size) )

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
    
    def realize_theta(self, seed=0, theta=None):
        self.x2_cache = {}

        if theta is not None:
            self.real_theta = theta
            self.theta = copy.deepcopy(self.real_theta)
            return self.theta
        
        self.real_theta = self.generate_theta_samples(self.real_delta, 1, seed=seed)[0]
        self.theta = copy.deepcopy(self.real_theta)
        return self.theta

    def x2star(self, excl: int = -1):
        # Here, we solve the second-stage market problem using Gurobipy, and return the optimal second-stage decision.
        if excl in self.x2_cache.keys():
            return self.x2_cache[excl]
        
        x1 = self.x1star(excl=excl)
        n = self.n

        model = gp.Model("Second-stage optimization")
        model.setParam('OutputFlag', 0)
        x2n = model.addVars(n, ub=self.D, name="x2") # final dispatch for uncertain sources
        x2n1 = model.addVar(lb = - GRB.INFINITY, name="x2n1") # reserve activation
        y2 = model.addVar(name='y2') # auxiliary variable representing abs(x2n1)
        x2n2 = model.addVar(name="x2n2") # load shedding
        # z2 = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='z2')
        c = model.addVars(n, name="c") # cost variables for producers
        
        model.setObjective( self.alpha_3 * y2 + self.alpha_4 * x2n2 + gp.quicksum( c[i] for i in range(n) ), GRB.MINIMIZE )
        
        [model.addGenConstrPWL(x2n[i], c[i], [0, self.theta[i][1], self.D], [-self.theta[i][0]*self.theta[i][1], 0, self.theta[i][2]*(self.D - self.theta[i][1])]) for i in range(n)]
        model.addConstr( self.D - gp.quicksum(x2n[i] * x1[i] for i in range(n)) - x1[-1] - x2n1 - x2n2 == 0, name="balance" )
        model.addConstr( y2 <= x1[-2] )
        model.addConstr( y2 >= x2n1 )
        model.addConstr( y2 >= -x2n1 )


        model.addConstrs( x2n[i] - self.theta[i][1] <= 15 for i in range(n) )
        model.addConstrs( self.theta[i][1] - x2n[i] <= 15 for i in range(n) )

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

    def average_outcomes(self, plot: bool = False):
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
        regulations = []

        theta_samples = self.generate_theta_samples(self.real_delta, self.batch_size)
        for j in range(self.batch_size):
            theta_sample = theta_samples[j]
            self.realize_theta(seed = 0, theta=theta_sample)
            x2 = self.second_stage_outcome()
            producer_regulation = list(np.array(x2[:-2]) - np.array([theta_sample[i][1] for i in range(self.n)]))
            payment2s.append([self.t2(i) for i in range(self.n)])
            payments.append( [payment1[i] + payment2s[j][i] for i in range(self.n)] )
            producer_costs.append( [self.ci(theta_sample[i], x1, x2, i) for i in range(self.n)] )
            utilities.append( [payments[j][i] - producer_costs[j][i] for i in range(self.n)] )
            reserve_activations.append(x2[-2])
            load_sheds.append(x2[-1])
            dispatch2s.append(x2[:-2])
            system_costs.append(self.system_cost(x1, x2, theta_sample))
            regulations.append(producer_regulation)


        metrics['average_utility'] = list(np.mean(utilities, axis=0))
        metrics['average_system_cost'] = np.mean(system_costs)
        metrics['payment1'] = payment1
        metrics['payment2'] = list(np.mean(payment2s, axis=0))
        metrics['payment2std'] = list(np.std(payment2s, axis=0))

        
        if plot:
            fig1 = plot_average_deviation(np.arange(1, self.n+1), producer_costs, xlabel='Producer', ylabel='Cost', title='Average Producer Costs with Std Dev')
            fig2 = plot_average_deviation([str(i+1) + 'x' if x1[i] <= 0 else str(i+1) for i in range(self.n)], [payment1, payment2s, payments], xlabel='Producer', ylabel='Payment', title='Average Payments with Std Dev', legends=['First-stage payment', 'Second-stage payment', 'Total payment'])
            fig3 = plot_average_deviation(np.arange(1, self.n+1), utilities, xlabel='Producer', ylabel='Utility', title='Average Utilities with Std Dev')
            fig4 = plot_average_deviation([1], reserve_activations, xlabel='', ylabel='Reserve Activation', title=f'Reserve Activations (Procured: {metrics["reserve_capacity"]:.2f}, Dispatchable: {metrics["dispatchable_power"]:.2f})')
            fig5 = plot_average_deviation([1], load_sheds, xlabel='', ylabel='Load Shedding', title='Load Shedding')
            fig6 = plot_average_deviation([1], system_costs, xlabel='', ylabel='System Cost', title='System Costs')
            fig7 = plot_average_deviation(np.arange(1, self.n+1), regulations, xlabel='Producer', ylabel='Regulation (Dispatch - Realized)', title='Producer Regulation Amounts with Std Dev')

            fig1.savefig('producer_costs.pdf')
            fig2.savefig('payments.pdf')
            fig3.savefig('utilities.pdf')
            fig4.savefig('reserve_activations.pdf')
            fig5.savefig('load_sheds.pdf')
            fig6.savefig('system_costs.pdf')
            fig7.savefig('regulations.pdf')

        return metrics

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
    
def utility_on_lying():
    market = StochasticMarket()
    utility = []
    delta = copy.deepcopy(market.real_delta)
    sigma_values = [0, 2, 4, 6, 8, 12, 20, 32] # standard deviation

    fig = plt.figure()
    plt.rcParams.update({'font.size': 14})
    ax = fig.add_subplot(1,1,1)

    for i in range(market.n):
        utility.append([])
        for s in tqdm(sigma_values):
            delta[i][1][1] = s
            market.update_params({'delta': delta})
            metrics = market.average_outcomes()
            utility[i].append( metrics['average_utility'][i] )
        ax.semilogx(np.power(sigma_values, 2), utility[i], label=f'Participant {i+1}')
        ax.plot(market.real_delta[i][1][1]**2, utility[i][sigma_values.index(market.real_delta[i][1][1])], marker='o', color='r') # mark true value
        delta[i][1][1] = market.real_delta[i][1][1] # reset to true value for next participant
    # ax.vlines(market.real_delta[i][1][1], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='r', linestyle='--', label='True sigma')
    ax.set_xlabel(f'Reported sigma by participants')
    ax.set_ylabel('Average Utility Received')
    ax.set_title(f'Impact of Lying on Participants\' Utility')
    ax.legend()
    fig.savefig(f'IC.pdf')
    return np.round(utility, 2)

def stochastic_vs_deterministic():
    # so in deterministic, participants will lie but their real sigma will remain the same
    market_stochastic = StochasticMarket()
    market_deterministic = StochasticMarket()
    delta = copy.deepcopy(market_deterministic.real_delta)
    
    stochastic_metrics = market_stochastic.average_outcomes()
    print("stochastic:")
    print("Dispatchable: ", stochastic_metrics['dispatchable_power'])
    print("Reserve: ", stochastic_metrics['reserve_capacity'])
    print("Average system cost: ", stochastic_metrics['average_system_cost'])

    deterministic_metrics = {}
    for d in [2, 10]:
        for i in range(market_deterministic.n):
            delta[i][1][1] = d
        market_deterministic.update_params({'delta': delta}) # participants report zero uncertainty in deterministic mechanism
        deterministic_metrics[d] = market_deterministic.average_outcomes()

        print(f"deterministic {d}:")
        print("Dispatchable: ", deterministic_metrics[d]['dispatchable_power'])
        print("Reserve: ", deterministic_metrics[d]['reserve_capacity'])
        print("Average system cost: ", deterministic_metrics[d]['average_system_cost'])
    
    json.dump({'stochastic': stochastic_metrics, 'deterministic': deterministic_metrics}, open('stochastic_vs_deterministic.json', 'w'), indent=4)

    return None

def payments_uncertainty():
    market = StochasticMarket(delta = [[[-100, 20, 300], [0, 2, 0]],
                                      [[-100, 20, 300], [0, 4, 0]],
                                      [[-100, 20, 300], [0, 6, 0]],
                                      [[-100, 20, 300], [0, 8, 0]],
                                      [[-100, 20, 300], [0, 32, 0]]])
    
    metrics = market.average_outcomes()
    sigma_values = [2, 4, 6, 8, 32]
    x = np.arange(len(sigma_values))
    payment1_all = [metrics['payment1']]
    payment2_all = [metrics['payment2']]
    payment2std_all = [metrics['payment2std']]

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.6
    ax.bar(x, metrics['payment1'], width, label='Payment 1', color='green', alpha=0.7)
    ax.bar(x, metrics['payment2'], width, bottom=metrics['payment1'], label='Payment 2', color='red', alpha=0.7)
    ax.errorbar(x, np.array(metrics['payment1']) + np.array(metrics['payment2']), yerr=metrics['payment2std'], fmt='o', color='black', capsize=5, label='Total')
    ax.set_xticks(x)
    ax.set_xticklabels([f'P{i+1}' for i in range(market.n)])
    ax.set_ylabel('Payment')
    ax.legend()
    fig.savefig('payments_unc.pdf')

def payments_flex():
    market = StochasticMarket(delta = [[[-10, 20, 100], [0, 10, 0]],
                                      [[-15, 20, 100], [0, 10, 0]],
                                      [[-20, 20, 100], [0, 10, 0]],
                                      [[-5, 20, 100], [0, 10, 0]],
                                      [[-2, 20, 100], [0, 10, 0]]])
    
    metrics = market.average_outcomes()
    flex_costs = [-10, -15, -20, -5, -2]
    x = np.argsort(flex_costs)
    payment1_all = [metrics['payment1']]
    payment2_all = [metrics['payment2']]
    payment2std_all = [metrics['payment2std']]

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.6
    ax.bar(x, metrics['payment1'], width, label='Payment 1', color='green', alpha=0.7)
    ax.bar(x, metrics['payment2'], width, bottom=metrics['payment1'], label='Payment 2', color='red', alpha=0.7)
    ax.errorbar(x, np.array(metrics['payment1']) + np.array(metrics['payment2']), yerr=metrics['payment2std'], fmt='o', color='black', capsize=5, label='Total')
    ax.set_xticks(x)
    ax.set_xticklabels([f'P{i+1}' for i in range(market.n)])
    ax.set_ylabel('Payment')
    ax.legend()
    fig.savefig('payments_flex.pdf')


if __name__ == "__main__":
    np.random.seed(0)
    
    print(utility_on_lying()) # Fig. 2 saves as IC.pdf
    payments_uncertainty() # Fig. 3 (top) saves as payments_unc.pdf
    payments_flex() # Fig. 3 (bottom) saves as payments_flex.pdf
    stochastic_vs_deterministic() # Table 1 results saved in stochastic_vs_deterministic.json