import os
import sys
from statsmodels.tsa.stattools import adfuller, acf, pacf
import numpy as np
from scipy import ndimage

## Calculate ARIMA parameters based on ACF and PACF plots
def calculate_arima_params(x, nlag = 50):
    lag_acf, confint_acf = acf(x[0:5000, 0], nlags=nlag, alpha=0.05)
    lag_pacf, confint_pacf = pacf(x[0:5000, 0], nlags=nlag, alpha=0.05)

    # Calculate the error bars (difference between value and confidence limit)
    # The "blue region" boundary is roughly 1.96 / sqrt(N) for large N
    lower_conf_acf = confint_acf[:, 0] - lag_acf
    upper_conf_acf = confint_acf[:, 1] - lag_acf
    def find_cutoff(values, conf_int):
        # confidence interval is centered at values, so we check if 0 is within [lower, upper]
        # conf_int[:, 0] is lower bound, conf_int[:, 1] is upper bound
        significant_lags = []
        for i in range(1, len(values)): # Start at 1 because lag 0 is always correlation 1
            if (0 < conf_int[i, 0]) or (0 > conf_int[i, 1]):
                significant_lags.append(i)
            else:
                # Once we hit the first insignificant lag, we stop (conservative approach)
                break
        return significant_lags[-1] if significant_lags else 0

    suggested_q = find_cutoff(lag_acf, confint_acf)
    suggested_p = find_cutoff(lag_pacf, confint_pacf)

    print(f"Visual inspection suggests: p (AR) = {suggested_p}, q (MA) = {suggested_q}")


## Calculate auto/cross-correlation matrix with delay
def correlation_calculation(x, y, range_min = 0, range_max = 50, step = 1):
    delay_range = range(range_min, range_max, step)
    cross_corr =[]
    for i in range(0, len(delay_range), 1):
        x_i = x[0:min(len(x),len(y))]
        y_i = ndimage.shift(x[0:min(len(x),len(y))], delay_range[i])
        cross_corr.append(np.corrcoef(x_i,y_i))
    return delay_range, cross_corr