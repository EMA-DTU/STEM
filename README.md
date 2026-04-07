# Stochastic Mechanism Design Experiments

This repository contains code for reproducing the numerical experiments from the paper on stochastic mechanism design for electricity markets.

## Prerequisites

- Python 3.13x
- Install required packages: `pip install -r requirements.txt`
- **Gurobi License**: The code uses Gurobi for optimization. A valid Gurobi license is required. Obtain a free academic license or commercial license from [Gurobi's website](https://www.gurobi.com/) if you don't have one.

## Running the Experiments

Execute the main script:

```bash
python numexp.py
```

This will run the experiments and generate the figures and data files.

## Experiment Descriptions

### 1. Incentive compatibility (Fig. 2)
Run `utility_on_lying()` to analyze utility changes with misreported production variance. Saves plot as `IC.pdf`.

### 2. Impact of uncertainty and flexibility on payments (Fig. 3)
Run `payments_uncertainty()` to analyze impact of production uncertainty on payments. Saves plot as `payments_unc.pdf`.
Run `payments_flex()` to analyze impact of production uncertainty on flexibility. Saves plot as `payments_flex.pdf`.

### 3. Stochastic vs Deterministic Mechanisms (Table 1)
Run `stochastic_vs_deterministic()` to compare stochastic vs. deterministic mechanisms. Saves results to `stochastic_vs_deterministic.json`.