import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import quandl
import fredapi
import os

from datetime import date
from pandas_datareader import data, fred
import pandas_datareader as pdr

yf.pdr_override()

# Set the start and end dates
start_date = pd.Timestamp('2013-01-01')
end_date = pd.Timestamp("2023-01-01")

def main_test():
    from pandas_datareader import data

    # Get a list of S&P 500 tickers
    tickers = pd.read_csv(r"Models/mean-variance-optimization/company-tickers-output.csv")["Ticker"].tolist()

    # Select a random sample of 10 tickers
    tickers = np.random.choice(tickers, 10, replace=False)

    print("Sample tickers:", tickers)

    # Fetch the stock data from Yahoo Finance for the selected tickers
    sample_data = None
    while sample_data is None or len(sample_data.columns) != len(tickers):
        try:
            sample_data = data.get_data_yahoo(list(tickers), start_date, end_date)["Adj Close"]
        except Exception as e:
            print("Error occurred:", e)
            print("Retrying...")

    print("Sample data:")
    df = pd.DataFrame(sample_data)
    print(df)

    fred_data = pdr.DataReader("FEDFUNDS", "fred", start_date, "today")
    rf = fred_data["FEDFUNDS"].tail(1).values[0]
    rf = int(rf) / 100

    # Calculate the covariance matrix and correlation matrix
    cov_matrix = df.pct_change().apply(lambda x: np.log(1 + x)).cov()
    corr_matrix = df.pct_change().apply(lambda x: np.log(1 + x)).corr()

    # Calculate the annualized individual returns
    ind_er = df.resample("Y").last().pct_change().mean()

    # Calculate the annualized standard deviation
    ann_sd = (
        df.pct_change()
        .apply(lambda x: np.log(1 + x))
        .std()
        .apply(lambda x: x * np.sqrt(250))
    )

    # Combine individual returns and standard deviations into a DataFrame
    assets = pd.concat([ind_er, ann_sd], axis=1)
    assets.columns = ["Returns", "Risk"]

    # Calculate the Sharpe ratio for each asset
    sharpe_ratio = (ind_er - rf) / ann_sd

    # Find the asset with the maximum Sharpe ratio
    max_sharpe_idx = sharpe_ratio.idxmax()

    # Calculate the optimal weights for the maximum Sharpe ratio asset
    optimal_weights = (
        cov_matrix.loc[:, max_sharpe_idx] / cov_matrix.loc[:, max_sharpe_idx].sum()
    )

    p_ret = []
    p_vol = []
    p_weights = []

    num_assets = len(df.columns)
    num_portfolios = 10000

    # Generate random portfolios
    for portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights = weights / np.sum(weights)
        p_weights.append(weights)
        returns = np.dot(weights, ind_er)
        p_ret.append(returns)
        var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
        sd = np.sqrt(var)
        ann_sd = sd * np.sqrt(250)
        p_vol.append(ann_sd)

    data = {"Returns": p_ret, "Volatility": p_vol}

    # Add weights for each asset to the data dictionary
    for counter, symbol in enumerate(df.columns.tolist()):
        data[symbol + " weight"] = [w[counter] for w in p_weights]

    # Create a DataFrame of the portfolio
    portfolios = pd.DataFrame(data)

    # Find the portfolio with the minimum volatility
    min_vol_port = portfolios.iloc[portfolios["Volatility"].idxmin()]

    # Find the portfolio with the optimal risky asset allocation
    optimal_risky_port = portfolios.iloc[
        ((portfolios["Returns"] - rf) / portfolios["Volatility"]).idxmax()
    ]

    # Plot the efficient frontier and optimal portfolios
    plt.figure(figsize=(10, 6))
    plt.scatter(
        portfolios["Volatility"], portfolios["Returns"], marker="o", s=10, alpha=0.3
    )
    plt.scatter(min_vol_port[1], min_vol_port[0], color="r", marker="*", s=500)
    plt.scatter(
        optimal_risky_port[1], optimal_risky_port[0], color="g", marker="*", s=500)

    # Find the index of the portfolio with the maximum Sharpe ratio
    max_sharpe_idx = portfolios["Returns"].idxmax()

    # Find the corresponding volatility and return values
    max_sharpe_vol = portfolios.loc[max_sharpe_idx, "Volatility"]
    max_sharpe_ret = portfolios.loc[max_sharpe_idx, "Returns"]

    # Plot the point for the portfolio with the maximum Sharpe ratio
    plt.scatter(
        max_sharpe_vol,
        max_sharpe_ret,
        color="yellow",
        marker="*",
        s=500,
    )

    plt.xlabel("Volatility")
    plt.ylabel("Returns")
    plt.title("Efficient Frontier and Optimal Portfolio")
    plt.legend(["Portfolios", "Minimum Volatility", "Optimal Risky", "Optimal Sharpe"])

    # Print the covariance matrix, correlation matrix, individual returns, and portfolio details
    print("\nCovariance matrix:\n", cov_matrix)
    print("\nCorrelation matrix:\n", corr_matrix)
    print("\nAnnualized individual returns:\n", ind_er)
    print("\nOptimal Risky Portfolio\n", optimal_risky_port)
    print("\nOptimal Sharpe Ratio Portfolio Weights:\n", optimal_weights)
    print("\nMinimum Volatility Portfolio\n", min_vol_port)

    # Display the plot
    plt.show()

if __name__ == "__main__":
    try:
        main_test()
    except:
        print("Exception occured, trying again...")
        main_test()