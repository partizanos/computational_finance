print(123)


def binomial(N, S0, s, r, K):
    tree = []
	return tree


binomial(N, S0, s, r, K)

S0 = 100
# initial time
t0 = 0
# time until option experiance,
call_maturiy_T = 1;t = call_maturiy_T
#steady; risk-free-rate
r=0.05
# K = Options striking pruce
K = 120
s = 0.2
max_tree_depth = 4

tree_depths = [ x for x in range(1, max_tree_depth)] 
call_estimated_values = [
	 binomial_model(N, S0, s, r, K)
	 for N in tree_depths
]