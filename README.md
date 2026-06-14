# Value at Risk Analysis using real financial data

## Overview

Value at Risk (VaR) is the standard risk measure used by banks, hedge funds, and regulators
worldwide. Under the Basel accords, financial institutions are required to compute and report VaR as part of their risk management framework. This project implements and compares three classical methods for VaR estimation using real financial data, and extends the analysis to Expected Shortfall. The three different methods of VaR calculation demonstrated in this project are the following:

- **Historical simulation** 

- **Parametric method**

- **Monte Carlo Simulation**

## Project Structure 

- **VaR.ipynb** - Value at Risk computation with different methods, figures comparing them and conclusions.

- **VaR_utils.py** - Module that contains modular, reusable functions for Value at Risk calculation and results comparison.

## Technologies Used

- Python 3.13.9
- Numpy
- Matplotlib
- Pandas
- Yfinance
- Jupyter Notebooks

## Skills Demonstrated 

- **Statistical Modeling**: Value at Risk computation based on different statistical assumptions
- **Monte Carlo Simulation**: Random generated experiments to simulate the daily returns distribution
- **Software Engineering**: Modular, reproducible Python code
- **Visualization**: Clear presentation and comparison of different methods
- **Quantitative Reasoning**: Extraction of conclusions based on each method used based on statistical principles
- **Financial Risk Analysis**: Implementation of industry-standard risk measures

## Usage

To run this project, the user needs to clone the repository and install the required libraries using the following command:

```bash
pip install numpy pandas matplotlib yfinance jupyter 
```

After having successfully installed the needed libraries, they can run the `VaR.ipynb` notebook using the following command:

```bash
jupyter notebook VaR.ipynb
```

## Value at Risk analysis - Key results

This project implements three different methods for the computation of the VaR. The different values computed are the following:

- **Historical Simulation VaR (99%): 5.37%**
- **Parametric VaR (99%): 4.54%**
- **Monte Carlo VaR (99%): 4.56%**

The conclusions that can be drawn from the different methods comparison are the following:

- The value calculated from the historical simulation is sitting noticeably further to the left compared to the values predicted by the other methods.

- The values predicted from the parametric and the Monte Carlo methods are almost overlapping with a difference of only 0.017%. This is justified given the assumption of a Gaussian distribution.

- The gap between the historical method and the parametric and Monte Carlo (0.83%) one can be entirely explained by the fat tails that are underestimated with the Gaussian distribution assumption.

## Expected Shortfall analysis - Key results

This project also extends the analysis to calculating the Expected Shortfall. The value computed is the following:

- **Expected Shortfall (99%): 7.41%**

Compared to the Value at Risk analysis, it suggests that:

- On the worst 1% of days, the average loss was 7.41%, which is 37.9% larger than the VaR threshold of 5.37%. This gap is caused by the extreme tails that are visible in the far left tail of the distribution.

- The Expected Shortfall measure constitutes a more accurate risk measure as it quantifies the severity of the loss on the worst 1% of days.