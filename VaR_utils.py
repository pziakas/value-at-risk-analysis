import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from scipy import stats

def get_returns(ticker: str, start_date: str, end_date: str) -> pd.Series:

    """
    Downloads historical price data for a given ticker and computes
    daily log returns.
    
    Parameters
    ----------
    ticker : str
        The stock ticker symbol (e.g. 'AAPL' for Apple).
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.
    
    Returns
    -------
    pd.Series
        Daily log returns for the given ticker.
    """

    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if data.empty:
        raise ValueError(f"No data found for ticker {ticker}. Check the ticker symbol and inserted dates!")
    
    prices = data["Close"][ticker]

    # Drop empty values

    returns = np.log(prices/prices.shift(1)).dropna()

    return returns

def plot_returns(returns: pd.Series, ticker: str, nbins: int = 100)-> None:
    """
    Plots the distribution of daily log returns and overlays a normal
    distribution with the same mean and standard deviation.
    
    Parameters
    ----------
    returns : pd.Series
        Daily log returns.
    ticker : str
        The stock ticker symbol, used for the plot title.
    nbins : int
        Number of histogram bins.
    """

    if nbins <= 0:
        raise ValueError("Number of bins must be positive!")
    
    mu = returns.mean()
    sigma = returns.std()

    _, ax = plt.subplots(figsize=(10, 6))

    ax.hist(returns, bins=nbins, density=True, histtype="step", color="steelblue", label="Empirical returns")

    x = np.linspace(returns.min(), returns.max(), 1000)

    normal_pdf = (1.0/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x-mu)/sigma) ** 2)

    ax.plot(x,normal_pdf,color="red", linewidth=2, label="Normal distribution")

    ax.set_xlabel("Daily log return")
    ax.set_ylabel("Density")
    ax.set_title(f"Return distribution for {ticker} (2018-2023)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Mean daily return: {mu:.6f}")
    print(f"Standard deviation of daily returns: {sigma:.6f}")
    print(f"Skewness: {returns.skew():.4f}")
    print(f"Kurtosis: {returns.kurtosis():.4f}")


def var_historical(returns: pd.Series, confidence: float) -> float:
    """
    Computes Value at Risk using historical simulation.
    
    Takes the empirical return distribution and reads off the loss
    at the given confidence level directly from the data, making no
    distributional assumptions.
    
    Parameters
    ----------
    returns : pd.Series
        Daily log returns.
    confidence : float
        Confidence level (e.g. 0.99 for 99% VaR).
    
    Returns
    -------
    float
        VaR expressed as a positive number representing a loss.
    """

    if confidence > 1.0 and confidence < 0.0:
        raise ValueError("The confidence level should be between 0 and 1")
    
    if len(returns) == 0:
        raise ValueError("The returns series is empty!")
    
    var = -np.percentile(returns,100*(1.0-confidence))

    return var

def var_parametric(returns: pd.Series, confidence: float) -> float:
    """
    Computes Value at Risk using the parametric (variance-covariance) method.
    
    Assumes returns are normally distributed and computes VaR analytically
    from the estimated mean and standard deviation.
    
    Parameters
    ----------
    returns : pd.Series
        Daily log returns.
    confidence : float
        Confidence level (e.g. 0.99 for 99% VaR).
    
    Returns
    -------
    float
        VaR expressed as a positive number representing a loss.
    """

    if confidence > 1.0 and confidence < 0.0:
        raise ValueError("The confidence level should be between 0 and 1")
    
    if len(returns) == 0:
        raise ValueError("The returns series is empty!")
    
    mu = returns.mean()
    sigma = returns.std()

    z = stats.norm.ppf(1-confidence)

    var = -(mu + sigma*z)

    return var

def var_montecarlo(returns: pd.Series, confidence: float, n_sim: int) -> float:
    """
    Computes Value at Risk using Monte Carlo simulation.
    
    Simulates a large number of possible one-day returns by drawing
    from a normal distribution fitted to the historical data, then
    reads off the loss at the given confidence level from the
    simulated distribution.
    
    Parameters
    ----------
    returns : pd.Series
        Daily log returns.
    confidence : float
        Confidence level (e.g. 0.99 for 99% VaR).
    n_simulations : int
        Number of simulated return scenarios.
    
    Returns
    -------
    float
        VaR expressed as a positive number representing a loss.
    """

    if confidence > 1.0 and confidence < 0.0:
        raise ValueError("The confidence level should be between 0 and 1")
    
    if len(returns) == 0:
        raise ValueError("The returns series is empty!")
    
    if n_sim <= 0:
        raise ValueError("The number of simulations must be positive!")
    
    mu = returns.mean()
    sigma = returns.std()

    np.random.seed(42)

    returns_sim = np.random.normal(mu, sigma, n_sim)

    var = -np.percentile(returns_sim,100*(1-confidence))

    return var


def plot_var_comparison(returns: pd.Series, var_hist: float, var_par: float, var_mc: float, ticker: str, confidence: float, nbins: int) -> None:
    """
    Plots the empirical return distribution with the three VaR estimates
    marked as vertical lines.
    
    Parameters
    ----------
    returns : pd.Series
        Daily log returns.
    var_hist : float
        VaR from historical simulation.
    var_param : float
        VaR from parametric method.
    var_mc : float
        VaR from Monte Carlo simulation.
    confidence : float
        Confidence level used for VaR computation.
    ticker : str
        Stock ticker symbol for the plot title.
    nbins : int
        Number of histogram bins.
    """

    if nbins <= 0:
        raise ValueError("The number of bins must be positive")
    
    if confidence > 1.0 and confidence < 0.0:
        raise ValueError("The confidence level should be between 0 and 1")
    
    _, ax = plt.subplots(figsize=(12, 6))

    ax.hist(returns, bins=nbins, density=True, histtype="step",color="steelblue", label="Empirical returns")

    mu = returns.mean()
    sigma = returns.std()


    x = np.linspace(returns.min(), returns.max(), 1000)

    normal_pdf = (1.0/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x-mu)/sigma) ** 2)

    ax.plot(x,normal_pdf,color="gray", linewidth=2, label="Normal distribution")

    ax.axvline(-var_hist, color="red", linewidth=2, label=f"Historical VaR ({confidence*100:.0f}%): {var_hist*100:.2f}%")
    ax.axvline(-var_par, color="orange", linewidth=2, label=f"Parametric VaR ({confidence*100:.0f}%): {var_par*100:.2f}%")
    ax.axvline(-var_mc, color="green", linewidth=1.5, linestyle="--", label=f"Monte Carlo VaR ({confidence*100:.0f}%): {var_mc*100:.2f}%")

    #ax.axhline(5.0,color="blue",linewidth=2,label="test")

    ax.set_xlabel("Daily log return")
    ax.set_ylabel("Density")
    ax.set_title(f"VaR comparison for {ticker} (2018-2023) - {confidence*100:.0f}% confidence")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n{'Method':<25} {'VaR':>10}")
    print("-" * 36)
    print(f"{'Historical Simulation':<25} {var_hist*100:>9.2f}%")
    print(f"{'Parametric':<25} {var_par*100:>9.2f}%")
    print(f"{'Monte Carlo':<25} {var_mc*100:>9.2f}%")
    print("-" * 36)
    print(f"{'Gap (Hist - Param)':<25} {(var_hist - var_par)*100:>9.2f}%")


def expected_shortfall(returns: pd.Series, confidence: float) -> float:
    """
    Computes Expected Shortfall (CVaR) using historical simulation.
    
    Expected Shortfall is the average loss on days where the loss
    exceeds the VaR threshold. It is a more informative risk measure
    than VaR as it captures the severity of losses in the tail.
    
    Parameters
    ----------
    returns : pd.Series
        Daily log returns.
    confidence : float
        Confidence level (e.g. 0.99 for 99% VaR).
    
    Returns
    -------
    float
        Expected Shortfall expressed as a positive number.
    """

    if confidence > 1.0 and confidence < 0.0:
        raise ValueError("The confidence level should be between 0 and 1")
    
    if len(returns) ==0:
        raise ValueError("The returns series is empty!")
    
    var = var_historical(returns, confidence)

    tail_losses = returns[returns < -var]

    es = - tail_losses.mean()

    return es

def plot_expected_shortfall(returns: pd.Series, var: float, es: float, confidence: float, ticker: str, nbins: int) -> None:
    """
    Plots the tail of the return distribution with VaR and Expected
    Shortfall marked. The shaded region represents the losses that
    contribute to the Expected Shortfall calculation.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns.
    var : float
        Value at Risk (historical simulation).
    es : float
        Expected Shortfall.
    confidence : float
        Confidence level used for computation.
    ticker : str
        Stock ticker symbol for the plot title.
    nbins : int
        Number of histogram bins.
    """

    if nbins <= 0:
        raise ValueError("The number of bins must be positive")
    
    if confidence > 1.0 and confidence < 0.0:
        raise ValueError("The confidence level should be between 0 and 1")
    
    _, ax = plt.subplots(figsize=(12, 6))
    
    counts, bin_edges, patches = ax.hist(returns, bins=nbins, density=True, histtype="stepfilled", color="steelblue", alpha=0.4, label="Empirical returns")

    for patch,left_edge in zip(patches, bin_edges[:-1]):
        if left_edge < -var:
            patch.set_facecolor("red")
            patch.set_alpha(0.6)


    ax.axvline(-var, color="red", linewidth=2, label=f"Historical VaR ({confidence*100:.0f}%): {var*100:.2f}%")
    ax.axvline(-es, color="darkred", linewidth=2, linestyle="--", label=f"Expected Shortfall ({confidence*100:.0f}%): {es*100:.2f}%")

    ax.set_xlabel("Daily log return")
    ax.set_ylabel("Density")
    ax.set_title(f"VaR and Expected Shortfall for {ticker} (2018-2023) - {confidence*100:.0f}% confidence")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax.set_xlim(returns.min() - 0.01, 0.01)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.show()

    print(f"\nNumber of days in the tails: {len(returns[returns < -var])}")
    print(f"Fraction of total days: {len(returns[returns < -var])/len(returns)}")