import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

# Load the data from the Excel file into a pandas dataframe
df_inputs = pd.read_excel('inputs.xlsx', sheet_name='inputs')
df_delta_scenarios = pd.read_excel('D:\client\not confirmed\nuc77\Delta_weight_scenarios (1).xlsx', sheet_name='Delta scenarios')

# Define the variables and functions required to calculate the PnL
spot_price = df_inputs['Spot Price'][0]
strike_price = df_inputs['Strike Price'][0]
time_to_expiry = df_inputs['Time to Expiry'][0]
risk_free_rate = df_inputs['Risk-free Rate'][0]
volatility = df_inputs['Volatility'][0]

def d1(S, K, r, sigma, T):
    return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

def d2(S, K, r, sigma, T):
    return d1(S, K, r, sigma, T) - sigma*np.sqrt(T)

def call_option_price(S, K, r, sigma, T):
    return S*norm.cdf(d1(S, K, r, sigma, T)) - K*np.exp(-r*T)*norm.cdf(d2(S, K, r, sigma, T))

def put_option_price(S, K, r, sigma, T):
    return K*np.exp(-r*T)*norm.cdf(-d2(S, K, r, sigma, T)) - S*norm.cdf(-d1(S, K, r, sigma, T))

def calculate_pnl(option_type, option_price, delta):
    if option_type == 'Call':
        pnl = (option_price - delta*(spot_price - strike_price)) - df_delta_scenarios['Call Price'][0]
    else:
        pnl = (option_price + delta*(spot_price - strike_price)) - df_delta_scenarios['Put Price'][0]
    return pnl

def maximize_pnl(x):
    call_delta, put_delta = x
    call_price = call_option_price(spot_price, strike_price, risk_free_rate, volatility, time_to_expiry)
    put_price = put_option_price(spot_price, strike_price, risk_free_rate, volatility, time_to_expiry)
    call_pnl = calculate_pnl('Call', call_price, call_delta)
    put_pnl = calculate_pnl('Put', put_price, put_delta)
    return -(call_pnl + put_pnl)

# Implement the optimization algorithm to maximize the PnL
x0 = [0.5, 0.5]
bounds = ((0, 1), (0, 1))
result = minimize(maximize_pnl, x0, bounds=bounds)
call_delta_optimized, put_delta_optimized = result.x

# Print the optimized values and the maximum PnL
print(f"Call Delta: {call_delta_optimized}")
print(f"Put Delta: {put_delta_optimized}")
max_pnl = -result.fun
print(f"Maximum PnL: {max_pnl}")
