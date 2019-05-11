# Let an asset S be valuated at t = 0 at S0 = 100. 
# Let be a call option 
# having maturity T = 1 and strike price K = 120. 
# The volatility is assumed to be constant (σ = 20%) over
# the lifespan of the call; the risk-free rate is r = 5%.
'''
		C = Call premium
		S = Current stock price
		t = time until option experiance,
		K = Options striking pruce
		r = Risk-free interest rate 
		N = cumulative standard normal distribution
		e = Exponential term
		s = St. Deviation 
		ln = natural log
'''
def Black_Scholes_Algo(
	C,
	S,
	t,
	K,
	r,
	N,
	e,
	s,
	ln):	
	
	d1 = ( 
		math.ln(S/K) + (r+s^2/2) *t) /(
		s * math.sqrt(t)
	)	
	d2 = d1 - s * math.sqrt(t)
 	
 	C = S * N(d1) - N(d2) * K*e**(-rt)
	return d1, d2, C

## assert 
# call_premium = 0 
C = 0 

# current stock price 
S0 = 100
S = S0
# initial time
t0 = 0

# time until option experiance,
call_maturiy_T = 1
t = t

#steady
# risk-free-rate
r=0.05

K = Options striking pruce
K = 120

# call_strike_price = 120
K = 120

volatility_sigma = 0.2


# 







#1.1
# '''
# The Black-Scholes formula (also called  Black-Scholes-Merton) 
# was the first widely used model for option pricing. 
# It's used to calculate the theoretical value of European-style options 
# using current stock prices, expected dividends, the option's strike price, 
# expected interest rates, time to expiration and expected volatility. 
# '''
# import math

# 	# underlying_price,
# 	# option_strike_price,
# 	# time_expiration_expressed_percent_year,
# 	# implied_volatility,
# 	# risk-free_interest_rates


# d1, d2, C = Black_Scholes_Algo
# print(d1, d2, C)
# #1.2

# def determine_value_call_t()
# 	return


# # binomial tree to determine the call option value.
# def determine_value_binomial_call_t(binomial_tree_depth)
# 	return

# call_estimated_value  = determine_value_call_t(t0)



# #1.3
# # Plot the evolution of the estimated value of the call option as a function of the
# # binomial tree depth.
# import maplot.pyplot as plt

# min_tree_depth = 1
# max_tree_depth = 3 
# binomial_tree_depth = [ x in range(min_tree_depth,max_tree_depth)]

# plt.plot(call_estimated_value, binomial_tree_depth)

# # 1.4 
# # How deep should be the tree in order to get a reasonable approximation?

# def reasonable_aproximatation_volatility():
# 	return ""
# reasonable_aproximatation = reasonable_aproximatation_volatility()