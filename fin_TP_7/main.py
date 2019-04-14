import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd


def libsConfig():
        np.random.seed(123)
        # Turn off progress printing
        solvers.options['show_progress'] = False

def getReturns():
        ''' closing prices of
         "Macdonald", "BankofAmerica", "IBM", "Chevron", "CocaCola", "Novartis", "ATT"
         #over a one-year time span extending from 2013-05-01 to 2014-05-01.'''
        stocks = pd.read_csv("closes.dat", sep="\t")
        stocks.columns = ["Macdonald", "BankofAmerica", "IBM", "Chevron", "CocaCola", "Novartis", "ATT"]
        # ret = np.log((stocks)/stocks.shift(1))
        ret = (stocks-stocks.shift(1))/stocks.shift(1)
        ret.drop([0], axis=0, inplace=True)
        return ret

def drawReturns(return_vec):
        plt.plot(return_vec.T, alpha=.4);
        plt.xlabel('time')
        plt.ylabel('returns')
        plt.show()

def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)


def random_portfolio(returns):
        '''
        Returns the mean and standard deviation of returns for a random portfolio
        '''

        p = np.asmatrix(np.mean(returns, axis=1))
        w = np.asmatrix(rand_weights(returns.shape[0]))
        C = np.asmatrix(np.cov(returns))

        mu = w * p.T
        sigma = np.sqrt(w * C * w.T)

        # This recursion reduces outliers to keep plots pretty
        if sigma > 2:
                return random_portfolio(returns)
        return mu, sigma

def drawMeanStd(stds, means):
        plt.plot(stds, means, 'o', markersize=5)
        plt.xlabel('std')
        plt.ylabel('mean')
        plt.title('Mean and standard deviation of returns of randomly generated portfolios')
        plt.show()


libsConfig()

ret = getReturns()
ret = ret.as_matrix()
returns = ret.T

n_assets = 7

## NUMBER OF OBSERVATIONS
n_obs = 251

# returns = np.random.randn(n_assets, n_obs)
drawReturns(returns)

n_portfolios = 500
portfolios = [
    random_portfolio(returns)
    for _ in range(n_portfolios)
]

(
        means,
        stds
)= np.column_stack(portfolios)

drawMeanStd(stds, means)


def optimal_portfolio(returns):
        n = len(returns)
        returns = np.asmatrix(returns)

        N = 100
        mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]

        # Convert to cvxopt matrices
        S = opt.matrix(np.cov(returns))
        pbar = opt.matrix(np.mean(returns, axis=1))

        # Create constraint matrices
        G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
        h = opt.matrix(0.0, (n, 1))
        A = opt.matrix(1.0, (1, n))
        b = opt.matrix(1.0)

        # Calculate efficient frontier weights using quadratic programming
        portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b)['x']
                      for mu in mus]
        ## CALCULATE RISKS AND RETURNS FOR FRONTIER
        returns = [blas.dot(pbar, x) for x in portfolios]
        risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
        ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
        m1 = np.polyfit(returns, risks, 2)
        x1 = np.sqrt(m1[2] / m1[0])
        # CALCULATE THE OPTIMAL PORTFOLIO
        wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
        return np.asarray(wt), returns, risks


weights, returns, risks = optimal_portfolio(returns)

plt.plot(stds, means, 'o')
plt.ylabel('mean')
plt.xlabel('std')
plt.plot(risks, returns, 'y-o')
plt.show()
