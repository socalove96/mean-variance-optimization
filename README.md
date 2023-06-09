# Mean-Variance Optimization Repository

This repository, located at [https://github.com/socalove96/mean-variance-optimization.git](https://github.com/socalove96/mean-variance-optimization.git), contains the code and files for my Mean-Variance Optimization project.

## Description

The Mean-Variance Optimization project focuses on implementing a portfolio optimization technique that aims to find the optimal allocation of assets in a portfolio based on their expected returns and variances. This technique allows investors to balance the trade-off between risk and return by constructing an efficient frontier of portfolios.

## Features

- **Mean-Variance Optimization Algorithm:** The repository includes the implementation of the mean-variance optimization algorithm, which calculates the optimal portfolio allocation based on expected returns and variances of assets.
- **Data Processing:** It provides functions for processing and preparing input data, such as historical returns and covariance matrices, required for the optimization algorithm.
- **Efficient Frontier Visualization:** The project includes tools for visualizing the efficient frontier, displaying the optimal portfolios that maximize returns for a given level of risk.
- **Performance Evaluation:** It offers functions to evaluate the performance of the optimized portfolios, including metrics such as Sharpe ratio, volatility, and portfolio weights.

## Installation

To set up and run this Mean-Variance Optimization project locally, please follow these steps:

1. Clone the repository using the following command:
   ```
   git clone https://github.com/socalove96/mean-variance-optimization.git
   ```

2. Navigate to the project directory:
   ```
   cd mean-variance-optimization
   ```

3. Install the required dependencies by running:
   ```
   pip install -r requirements.txt
   ```

4. Run the main script or explore the Jupyter notebooks provided to understand and utilize the Mean-Variance Optimization functionality.

# Mean-Variance Optimization Model
The provided code performs a mean-variance optimization for a portfolio of stocks using historical price data. It uses the pandas, numpy, and matplotlib libraries for data manipulation and visualization. The yfinance, pandas_datareader, and fredapi libraries are used to fetch stock and macroeconomic data.

The main_test() function is the entry point of the code. It reads a list of S&P 500 tickers from a CSV file and randomly selects a sample of 10 tickers. The historical adjusted close prices for these tickers are fetched from Yahoo Finance. It also retrieves the current risk-free rate from the FRED API.

The code then calculates the covariance matrix and correlation matrix for the stock returns. It calculates the annualized individual returns and standard deviations for each stock. These values are combined into a DataFrame called "assets." The Sharpe ratio is calculated for each stock to measure its risk-adjusted performance.

Next, the code generates 10,000 random portfolios by assigning random weights to the stocks. For each portfolio, it calculates the expected return, volatility, and weights of each stock. The data is stored in a DataFrame called "portfolios."

The code identifies the portfolio with the minimum volatility (min_vol_port) and the portfolio with the optimal risky asset allocation (optimal_risky_port). It also finds the portfolio with the maximum Sharpe ratio and its corresponding volatility and return values.

Finally, the efficient frontier and the optimal portfolios are plotted using matplotlib. The covariance matrix, correlation matrix, individual returns, and details of the optimal portfolios are printed. If any exceptions occur during the execution, the code attempts to run the main_test() function again.

Overall, this code implements a mean-variance optimization to construct efficient portfolios and visualize their risk-return characteristics.

# Limitations of the Code
**Limited data sources:** The code relies on Yahoo Finance and FRED API as data sources. This limits the availability of data to the stocks listed on Yahoo Finance and the macroeconomic indicators provided by FRED. There might be stocks or additional data sources that are not supported.

**Single optimization approach:** The code uses mean-variance optimization, which assumes that returns follow a normal distribution and that investors' preferences can be adequately captured by mean and variance. However, this approach has limitations, such as sensitivity to input parameters and the assumption of linear relationships. It does not account for factors like non-normal return distributions, transaction costs, and investor-specific constraints.

**Lack of risk modeling:** The code does not incorporate more sophisticated risk modeling techniques like factor models, time-varying volatility, or downside risk measures. It uses historical volatility as a measure of risk, which may not fully capture the future risk characteristics of the assets.

**No consideration of transaction costs:** The code does not account for transaction costs associated with portfolio rebalancing or trading. In practice, transaction costs can significantly impact portfolio performance and should be considered in the optimization process.

**Ignoring liquidity constraints:** The code assumes that all assets are equally liquid and that there are no constraints on trading. In reality, some assets may have limited liquidity, and there may be restrictions on trading certain stocks or asset classes.

**Lack of robustness testing:** The code does not perform robustness testing or sensitivity analysis to evaluate the stability and reliability of the optimized portfolios under different market conditions or input assumptions. It is important to assess the performance of the portfolios across various scenarios.

**Data limitations:** The code retrieves historical price data and calculates returns based on this data. It does not consider other important data points such as corporate actions (e.g., dividends, stock splits) or fundamental factors that could impact the stock returns. Incorporating additional data and preprocessing steps may improve the accuracy of the optimization results.

**Lack of out-of-sample testing:** The code does not reserve a separate dataset for out-of-sample testing. Evaluating the performance of optimized portfolios on unseen data can provide a better assessment of their effectiveness and robustness.

**Lack of portfolio constraints:** The code does not include any portfolio constraints, such as minimum or maximum allocation limits, sector diversification requirements, or constraints on leverage. These constraints are common in practice and should be considered when constructing portfolios.

## Contributing

Thank you for your interest in contributing to the Mean-Variance Optimization project! If you have any suggestions, bug reports, or improvements, please feel free to open an issue or submit a pull request. Your contributions are greatly appreciated.

## License

This repository is licensed under the [MIT License](https://opensource.org/licenses/MIT). Please refer to the `LICENSE` file for more details.

## Contact

If you have any questions or would like to reach out, please feel free to contact me via email at alex.malek96@gmail.com(mailto:alex.malek96@gmail.com).

Thank you for visiting the Mean-Variance Optimization repository!


