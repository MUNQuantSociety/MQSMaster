import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Example tickers and their historical mean returns and covariance matrix
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Dummy mean returns (annualized)
mean_returns = np.array([0.12, 0.10, 0.11, 0.13, 0.15])

# Dummy covariance matrix (annualized)
cov_matrix = np.array([
    [0.04, 0.006, 0.008, 0.005, 0.007],
    [0.006, 0.03, 0.006, 0.007, 0.005],
    [0.008, 0.006, 0.035, 0.006, 0.006],
    [0.005, 0.007, 0.006, 0.04, 0.008],
    [0.007, 0.005, 0.006, 0.008, 0.05],
])

MIN_WEIGHT = 0.05  # minimum weight 5%

def optimize_weights(risk_appetite):
    """
    Simple example: risk_appetite between 0 (low risk) and 1 (high risk)
    We'll allocate more weight to lower volatility stocks if risk_appetite is low,
    and more evenly if risk_appetite is high.
    """

    volatilities = np.sqrt(np.diag(cov_matrix))
    inv_vol = 1 / volatilities
    base_weights = inv_vol / inv_vol.sum()  # inverse volatility weighting
    
    # Adjust weights according to risk appetite
    # At risk_appetite=0, weights = base_weights (low risk)
    # At risk_appetite=1, weights = equal weights (higher risk)
    equal_weights = np.ones_like(base_weights) / len(base_weights)
    weights = (1 - risk_appetite) * base_weights + risk_appetite * equal_weights

    # Enforce minimum weight constraint (5%)
    weights = np.maximum(weights, MIN_WEIGHT)
    weights /= weights.sum()  # re-normalize to sum to 1

    return weights

def plot_weights(weights):
    fig = go.Figure(go.Bar(
        x=tickers,
        y=weights,
        marker_color='lightskyblue'
    ))
    fig.update_layout(
        title="Portfolio Weights by Ticker",
        yaxis_title="Weight",
        yaxis=dict(range=[0,1])
    )
    fig.show()

def plot_risk_return(weights):
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(weights.T @ cov_matrix @ weights)

    fig = go.Figure()

    # Plot the portfolio point
    fig.add_trace(go.Scatter(
        x=[port_vol], y=[port_return],
        mode='markers',
        marker=dict(size=12, color='red'),
        name='Portfolio'
    ))

    # Plot individual assets for reference
    volatilities = np.sqrt(np.diag(cov_matrix))
    fig.add_trace(go.Scatter(
        x=volatilities,
        y=mean_returns,
        mode='markers+text',
        text=tickers,
        textposition="top center",
        marker=dict(size=10, color='blue'),
        name='Assets'
    ))

    fig.update_layout(
        title='Portfolio Risk vs Return',
        xaxis_title='Volatility (Risk)',
        yaxis_title='Expected Return',
        xaxis=dict(range=[0, max(volatilities)*1.5]),
        yaxis=dict(range=[0, max(mean_returns)*1.5])
    )
    fig.show()

def main():
    print("Enter risk appetite (0=low risk, 1=high risk): ")
    risk_appetite = float(input().strip())
    if risk_appetite < 0 or risk_appetite > 1:
        print("Risk appetite must be between 0 and 1")
        return

    weights = optimize_weights(risk_appetite)
    print("\nCalculated portfolio weights:")
    for ticker, weight in zip(tickers, weights):
        print(f"{ticker}: {weight:.2%}")

    plot_weights(weights)
    plot_risk_return(weights)

if __name__ == "__main__":
    main()
