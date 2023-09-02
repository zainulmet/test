import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Simulated historical stock price data for three companies (AAPL, MSFT, GOOGL)
data = pd.read_csv('stock_data.csv', index_col='Date')
returns = data.pct_change().dropna()  # Calculate daily returns

# Define the expected annual return and covariance matrix of returns
expected_returns = returns.mean() * 252  # 252 trading days in a year
cov_matrix = returns.cov() * 252

# Define a function to calculate portfolio metrics
def portfolio_metrics(weights):
    portfolio_return = np.sum(expected_returns * weights)
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_stddev

# Define the optimization objective function (to minimize portfolio risk)
def objective_function(weights):
    return portfolio_metrics(weights)[1]  # Minimize portfolio standard deviation

# Define optimization constraints (sum of weights = 1)
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

# Define bounds for individual stock weights (0 to 1)
bounds = tuple((0, 1) for _ in range(len(returns.columns)))

# Initialize equal weights for all stocks
initial_weights = [1.0 / len(returns.columns) for _ in range(len(returns.columns))]

# Perform portfolio optimization
result = minimize(objective_function, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

# Extract the optimized portfolio weights
optimal_weights = result.x

# Calculate the expected return and standard deviation of the optimized portfolio
optimal_return, optimal_stddev = portfolio_metrics(optimal_weights)

# Visualize the efficient frontier
portfolio_returns = []
portfolio_stddevs = []

for _ in range(10000):
    random_weights = np.random.random(len(returns.columns))
    random_weights /= np.sum(random_weights)
    portfolio_return, portfolio_stddev = portfolio_metrics(random_weights)
    portfolio_returns.append(portfolio_return)
    portfolio_stddevs.append(portfolio_stddev)

plt.figure(figsize=(10, 6))
plt.scatter(portfolio_stddevs, portfolio_returns, c='b', marker='o', label='Random Portfolios')
plt.scatter(optimal_stddev, optimal_return, c='r', marker='x', label='Optimal Portfolio')
plt.title('Portfolio Optimization')
plt.xlabel('Portfolio Risk (Standard Deviation)')
plt.ylabel('Expected Portfolio Return')
plt.legend()
plt.grid(True)

# Print the results
print("Optimal Portfolio Weights:")
for i, symbol in enumerate(returns.columns):
    print(f"{symbol}: {optimal_weights[i]:.4f}")

print(f"Optimal Portfolio Expected Return: {optimal_return:.4f}")
print(f"Optimal Portfolio Risk (Standard Deviation): {optimal_stddev:.4f}")

# Save the plot as an image
plt.savefig('portfolio_optimization.png')

# Perform sensitivity analysis, scenario analysis, or other financial modeling tasks.
