import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import pylab
import yfinance as yf 
import numpy as np
import pandas as pd
import scipy.optimize
import warnings

warnings.filterwarnings('ignore')
axLblFont = font_manager.FontProperties(family='Calibri', weight='bold', style='normal', size=10)
lgndFont = font_manager.FontProperties(family='Calibri', weight='normal', style='normal', size=8)

def preprocess_ticker_list(ticker_list):
    string = str(ticker_list)
    #remove spaces
    string.replace(" ", "")
    #split string on commas
    return string.split(",")

def port_var(W, C):
    return np.dot(np.dot(W, C), W)

def port_mean_var(W, R, C):
    return sum(R * W), np.dot(np.dot(W, C), W)

def solve_frontier(R, C, rf):
    def fitness(W, R, C, r):
        # For given level of return r, find weights which minimizes portfolio variance.
        mean, var = port_mean_var(W, R, C)
        penalty = 100 * abs(mean - r)  # Big penalty for not meeting stated portfolio return 
        return var + penalty

    frontier_mean, frontier_var, frontier_weights = [], [], []
    n = len(R)  # Number of assets in the portfolio
    for r in np.linspace(min(R), max(R), num=40):  # Iterate through the range of y-axis returns
        W = np.ones([n]) / n  # initial value of weights = 1/n
        b_ = [(0, 1) for i in range(n)]
        c_ = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})
        optimized = scipy.optimize.minimize(fitness, W, (R, C, r), method='SLSQP', constraints=c_, bounds=b_, tol=1e-12, options={'maxiter': 10000})
        if not optimized.success:
            raise BaseException(optimized.message)
        # add point to the efficient frontier [x,y] = [optimized.x, r]
        frontier_mean.append(r)
        frontier_var.append(port_var(optimized.x, C))
        frontier_weights.append(optimized.x)
    return np.array(frontier_mean), np.array(frontier_var), frontier_weights

def solve_weights(R, C, rf):
    def fitness(W, R, C, rf):
        mean, var = port_mean_var(W, R, C)
        util = (mean - rf)/np.sqrt(var)  # Sharpe ratio
        return 1/util
    n = len(R)
    W = np.ones([n])/n
    b_ = [(0., 1.) for i in range(n)]  # No shorting
    c_ = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})  # Ensure weights add up to 1
    optimized = scipy.optimize.minimize(fitness, W, (R, C, rf), method='SLSQP', constraints=c_, bounds=b_)
    if not optimized.success: raise BaseException(optimized.message)
    return optimized.x

class Result:
    def __init__(self, W, tan_mean, tan_var, front_mean, front_var, front_weights):
        self.W=W
        self.tan_mean=tan_mean
        self.tan_var=tan_var
        self.front_mean=front_mean
        self.front_var=front_var
        self.front_weights=front_weights
        
def optimize_frontier(R, C, rf):
    W = solve_weights(R, C, rf)
    tan_mean, tan_var = port_mean_var(W, R, C)  # calculate tangency portfolio
    front_mean, front_var, front_weights = solve_frontier(R, C, rf)  # calculate efficient frontier
    return Result(W, tan_mean, tan_var, front_mean, front_var, front_weights)

def display_assets(names, R, C, color='black'):
    pylab.scatter([C[i, i] ** .5 for i in range(n)], R, marker='x', color=color), pylab.grid(True)
    for i in range(n): 
        pylab.text(C[i, i] ** .5, R[i], '  %s' % names[i], verticalalignment='center', color=color)

def display_frontier(result: Result, label=None, color='black'):
    pylab.text(result.tan_var ** .5, result.tan_mean, '   tangent', verticalalignment='center', color=color)
    pylab.scatter(result.tan_var ** .5, result.tan_mean, marker='o', color=color), pylab.grid(True)
    pylab.plot(list(result.front_var ** .5), list(result.front_mean), label=label, color=color), pylab.grid(True)    

def load_data(symbols, cap):
    #symbols = ["IXE","IXM","IXY","IXR","IXT"]
    #cap = {'IXE': 2.14e12, 'IXM': 6.86e12, 'IXY': 7.97e12, 'IXR': 4.04e12, 'IXT': 12.64e12}
    n = len(symbols)
    prices_out, caps_out = [], []
    for s in symbols:
        dataframe = pd.read_csv('data2/%s.csv' % s, index_col=None, parse_dates=['Date'])
        prices = list(dataframe['Close'])[-500:] # trailing window 500 days
        prices_out.append(prices)
        caps_out.append(cap[s])
    return symbols, prices_out, caps_out

def assets_historical_returns_and_covariances(prices):
    prices = np.matrix(prices)
    # create matrix of historical returns
    rows, cols = prices.shape
    returns = np.empty([rows, cols - 1])
    for r in range(rows):
        for c in range(cols - 1):
            p0, p1 = prices[r, c], prices[r, c + 1]
            returns[r, c] = (p1 / p0) - 1
    # calculate returns
    expreturns = np.array([])
    for r in range(rows):
        expreturns = np.append(expreturns, np.mean(returns[r]))
    # calculate covariances
    covars = np.cov(returns)
    expreturns = (1 + expreturns) ** 250 - 1  # Annualize returns
    covars = covars * 250  # Annualize covariances
    return expreturns, covars

def create_views_and_link_matrix(names, views):
    r, c = len(views), len(names)
    Q = [views[i][3] for i in range(r)]  # view matrix
    P = np.zeros([r, c])
    nameToIndex = dict()
    for i, n in enumerate(names):
        nameToIndex[n] = i
    for i, v in enumerate(views):
        name1, name2 = views[i][0], views[i][2]
        P[i, nameToIndex[name1]] = +1 if views[i][1] == '>' else -1
        P[i, nameToIndex[name2]] = -1 if views[i][1] == '>' else +1
    return np.array(Q), P

symbols = ["IXE","IXM","IXY","IXR","IXT"]
cap = {'IXE': 2.14e12, 'IXM': 6.86e12, 'IXY': 7.97e12, 'IXR': 4.04e12, 'IXT': 12.64e12}
names, prices, caps = load_data(symbols, cap)
ticker_list = symbols
n = len(names)

W = np.array(caps)/np.sum(caps) # calculate market weights from capitalizations
R, C = assets_historical_returns_and_covariances(prices)
rf = 0.015  # Risk-free rate

res1 = optimize_frontier(R, C, rf)
print("Done optimizing")
# Plot Markowitz bullet
sigP1 = list(res1.front_var ** 0.5)
muP1 = list(res1.front_mean)
sigM1 = res1.tan_var ** .5
muM1 = res1.tan_mean
wM1 = res1.W
sharpeRatio = (muM1 - rf)/sigM1
sharpeRatioText = 'Sharpe Ratio = ' + str(round(sharpeRatio, 3))
# Create text for displaying weights @ tangency point on Markowitz bullet
weightsText = 'Weights @ Tangency\n'
for i in range(len(wM1)):
    weightsText = weightsText + ticker_list[i].strip() + ' = ' + str(round(wM1[i], 3)) + '\n'
fig, ax = plt.subplots(ncols=1, figsize=(7.0,4.0), dpi=200)
fig.subplots_adjust(left=0.1, right=0.7, bottom=0.125)
ax.plot(sigP1, muP1, label='Effcient Frontier')
ax.scatter(sigM1, muM1, label='Tangency Portfolio')
ax.plot([0, (max(muP1) - rf)/sharpeRatio], [rf, max(muP1)], lw=1, label='Capital Market Line')
ax.legend()
handles, labels = ax.get_legend_handles_labels()
lgd = dict(zip(labels, handles))
lgnd = ax.legend(lgd.values(), lgd.keys(), bbox_to_anchor=(1.04,1), loc="upper left", prop=lgndFont)
ax.grid()
ax.text(0.2, 0.9, sharpeRatioText, size=8, ha="center", transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='w'))
ax.text(1.06, 0.75, weightsText, size=8, ha="left", va="top", linespacing=1.5, transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='w'))
xAxLabel = 'Standard Deviation (Risk)'
yAxLabel = 'Annual Portfolio Return'
ax.set_xlabel(xAxLabel, fontproperties=axLblFont, color='k')
ax.set_ylabel(yAxLabel, fontproperties=axLblFont, color='k', labelpad=2)
ax.tick_params(axis='both', which='major', labelsize=8)
ax.set_xlim([0, 1.1*max(sigP1)])
ax.set_ylim([0, 1.1*max(muP1)])
fig.suptitle('Portfolio Frontier', weight='bold', fontsize=10, x=0.4, y=0.95)
plt.show()

# Calculate portfolio historical return and variance
mean, var = port_mean_var(W, R, C)
lmb = (mean - rf) / var  # Calculate risk aversion
Pi = np.dot(np.dot(lmb, C), W)  # Calculate equilibrium excess returns
res2 = optimize_frontier(Pi+rf, C, rf)
print("Done optimizing")
# Plot Markowitz bullet
sigP2 = list(res2.front_var ** 0.5)
muP2 = list(res2.front_mean)
sigM2 = res2.tan_var ** .5
muM2 = res2.tan_mean
wM2 = res2.W
#plotFrontier(sigP2, muP2, sigM2, muM2, wM2)
sharpeRatio = (muM2 - rf)/sigM2
sharpeRatioText = 'Sharpe Ratio = ' + str(round(sharpeRatio, 3))
# Create text for displaying weights @ tangency point on Markowitz bullet
weightsText = 'Weights @ Tangency\n'
for i in range(len(wM2)):
    weightsText = weightsText + ticker_list[i].strip() + ' = ' + str(round(wM2[i], 3)) + '\n'
fig, ax = plt.subplots(ncols=1, figsize=(7.0,4.0), dpi=200)
fig.subplots_adjust(left=0.1, right=0.7, bottom=0.125)
ax.plot(sigP2, muP2, label='Implied Returns')
ax.plot(sigP1, muP1, label='Historical Returns', color='red', lw=0.5)
ax.scatter(sigM2, muM2, label='Tangency Portfolio')
ax.plot([0, (max(muP2) - rf)/sharpeRatio], [rf, max(muP2)], lw=1, label='Capital Market Line')
ax.legend()
handles, labels = ax.get_legend_handles_labels()
lgd = dict(zip(labels, handles))
lgnd = ax.legend(lgd.values(), lgd.keys(), bbox_to_anchor=(1.04,1), loc="upper left", prop=lgndFont)
ax.grid()
ax.text(0.2, 0.9, sharpeRatioText, size=8, ha="center", transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='w'))
ax.text(1.06, 0.75, weightsText, size=8, ha="left", va="top", linespacing=1.5, transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='w'))
xAxLabel = 'Standard Deviation (Risk)'
yAxLabel = 'Annual Portfolio Return'
ax.set_xlabel(xAxLabel, fontproperties=axLblFont, color='k')
ax.set_ylabel(yAxLabel, fontproperties=axLblFont, color='k', labelpad=2)
ax.tick_params(axis='both', which='major', labelsize=8)
ax.set_xlim([0, 1.1*max(sigP2)])
ax.set_ylim([0, 1.1*max(muP2)])
fig.suptitle('Portfolio Frontier', weight='bold', fontsize=10, x=0.4, y=0.95)
plt.show()

views = [('IXE', '>', 'IXM', 0.1),
         ('IXY', '<', 'IXT', 0.02)]

Q, P = create_views_and_link_matrix(names, views)
tau = .025  # scaling factor

# Calculate omega - uncertainty matrix about views
omega = np.dot(np.dot(np.dot(tau, P), C), np.transpose(P))  # 0.025*P*C*transpose(P)

# Calculate equilibrium excess returns with views incorporated
sub_a = np.linalg.inv(np.dot(tau, C))
sub_b = np.dot(np.dot(np.transpose(P), np.linalg.inv(omega)), P)
sub_c = np.dot(np.linalg.inv(np.dot(tau, C)), Pi)
sub_d = np.dot(np.dot(np.transpose(P), np.linalg.inv(omega)), Q)
Pi_adj = np.dot(np.linalg.inv(sub_a + sub_b), (sub_c + sub_d))
res3 = optimize_frontier(Pi_adj + rf, C, rf)
print("Done optimizing")
# Plot Markowitz bullet
sigP3 = list(res3.front_var ** 0.5)
muP3 = list(res3.front_mean)
sigM3 = res3.tan_var ** .5
muM3 = res3.tan_mean
wM3 = res3.W
#plotFrontier(sigP3, muP3, sigM3, muM3, wM3)
sharpeRatio = (muM3 - rf)/sigM3
sharpeRatioText = 'Sharpe Ratio = ' + str(round(sharpeRatio, 3))
# Create text for displaying weights @ tangency point on Markowitz bullet
weightsText = 'Weights @ Tangency\n'
for i in range(len(wM3)):
    weightsText = weightsText + ticker_list[i].strip() + ' = ' + str(round(wM3[i], 3)) + '\n'
fig, ax = plt.subplots(ncols=1, figsize=(7.0,4.0), dpi=200)
fig.subplots_adjust(left=0.1, right=0.7, bottom=0.125)
ax.plot(sigP3, muP3, label='Adjusted Returns')
ax.plot(sigP2, muP2, label='Implied Returns', color='red', lw=0.5)
ax.scatter(sigM3, muM3, label='Tangency Portfolio')
ax.plot([0, (max(muP3) - rf)/sharpeRatio], [rf, max(muP3)], lw=1, label='Capital Market Line')
ax.legend()
handles, labels = ax.get_legend_handles_labels()
lgd = dict(zip(labels, handles))
lgnd = ax.legend(lgd.values(), lgd.keys(), bbox_to_anchor=(1.04,1), loc="upper left", prop=lgndFont)
ax.grid()
ax.text(0.2, 0.9, sharpeRatioText, size=8, ha="center", transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='w'))
ax.text(1.06, 0.75, weightsText, size=8, ha="left", va="top", linespacing=1.5, transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='w'))
xAxLabel = 'Standard Deviation (Risk)'
yAxLabel = 'Annual Portfolio Return'
ax.set_xlabel(xAxLabel, fontproperties=axLblFont, color='k')
ax.set_ylabel(yAxLabel, fontproperties=axLblFont, color='k', labelpad=2)
ax.tick_params(axis='both', which='major', labelsize=8)
ax.set_xlim([0, 1.1*max(sigP3)])
ax.set_ylim([0, 1.1*max(muP3)])
fig.suptitle('Portfolio Frontier', weight='bold', fontsize=10, x=0.4, y=0.95)
plt.show()

"""
# Download data and create log return dataframe
#ticker_list = preprocess_ticker_list('IXM, IXY, IXR')
ticker_list = preprocess_ticker_list('IXE, IXM, IXY, IXR, IXT')
startdate = "2000-01-01"
enddate = "2020-01-01"
print(ticker_list[0])
data = yf.download(ticker_list[0], startdate, enddate, interval="1mo")
data.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
dataLogRet = np.log(data['Adj Close']) - np.log(data['Adj Close'].shift(1))
for ticker in ticker_list[1:]:
    print(ticker)
    temporData = yf.download(ticker, startdate, enddate, interval="1mo")
    temporData.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
    temporDataLogRet = np.log(temporData['Adj Close']) - np.log(temporData['Adj Close'].shift(1))
    data = pd.concat([data, temporData], axis=1)
    dataLogRet = pd.concat([dataLogRet, temporDataLogRet], axis=1)
data.columns = ticker_list
dataLogRet.columns = ticker_list
dataLogRet = dataLogRet[1:]
print(dataLogRet)

# Calculate mean return matrix (mu), covariance matrix (V) and unit column vector (e)
mu = np.zeros(len(ticker_list)).reshape(len(ticker_list), 1)
i = 0
for ticker in ticker_list:
    mu[i,0] = dataLogRet[ticker].mean()
    i = i+1
mu = 12 * mu
e = np.ones(len(ticker_list)).reshape(len(ticker_list), 1)
V = dataLogRet.cov()
V = 12 * 12 * V

#mu = np.array([[0.08], [0.12], [0.1]])
#e = np.array([[1.0], [1.0], [1.0]])
#V = np.array([[0.0081, -0.00675, 0], [-0.00675, 0.0225, 0.003], [0, 0.003, 0.01]])

# Set risk free rate (annual)
rRF = 0.01

# Calculate A, B, C values and determinant AC - Bsquared
Vinv = np.linalg.inv(V)
A = np.matmul(Vinv, e)
A = np.matmul(e.transpose(), A)
B = np.matmul(Vinv, e)
B = np.matmul(mu.transpose(), B)
C = np.matmul(Vinv, mu)
C = np.matmul(mu.transpose(), C)
denom = A*C - B*B

# Calculate weights, return, and risk at global minimum point
wG = np.matmul(Vinv, e)
wG = wG/A
muG = np.matmul(mu.transpose(), wG)[0,0]
sigG = 1/np.sqrt(A)[0,0]

# Calculate weights, return, and risk at tangency point (aka Market portfolio)
muM = (C - B*rRF)/(B - A*rRF)
muM = muM[0,0]
sigM = np.sqrt((A*muM*muM - 2*B*muM + C)/denom)[0,0]
wM = ((C - muM*B)/denom)*(np.matmul(Vinv, e)) + ((muM*A - B)/denom)*(np.matmul(Vinv, mu))
wM = wM[:,0]
sharpeRatio = (muM - rRF)/sigM
sharpeRatioText = 'Sharpe Ratio = ' + str(round(sharpeRatio, 3))

# Scan portfolio return (muP), compute risk (sigP) at each point of scan
upperLim = round((1.05 * muM/muG), 1)
lowerLim = max((2.0 - upperLim), 0.05)
scale = np.linspace(lowerLim, upperLim, 51)
#scale = np.linspace(0.25, 1.75, 51)
muP = muG*scale
sigP = np.sqrt((A*muP*muP - 2*B*muP + C)/denom)[0]

# Create text for displaying weights @ tangency point on Markowitz bullet
weightsText = 'Weights @ Tangency\n'
for i in range(len(wM)):
    weightsText = weightsText + ticker_list[i].strip() + ' = ' + str(round(wM[i], 3)) + '\n'

# Plot Markowitz bullet
fig, ax = plt.subplots(ncols=1, figsize=(7.0,4.0), dpi=100)
fig.subplots_adjust(left=0.1, right=0.7, bottom=0.125)
ax.plot(sigP, muP, label='Effcient Frontier')
ax.scatter(sigM, muM, label='Tangency Portfolio')
ax.plot([0, (max(muP) - rRF)/sharpeRatio], [rRF, max(muP)], lw=1, label='Capital Market Line')
ax.legend()
handles, labels = ax.get_legend_handles_labels()
lgd = dict(zip(labels, handles))
lgnd = ax.legend(lgd.values(), lgd.keys(), bbox_to_anchor=(1.04,1), loc="upper left", prop=lgndFont)
ax.grid()
ax.text(0.5, 0.05, sharpeRatioText, size=8, ha="center", transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='w'))
ax.text(1.06, 0.75, weightsText, size=8, ha="left", va="top", linespacing=1.5, transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='w'))
xAxLabel = 'Standard Deviation (Risk)'
yAxLabel = 'Annual Portfolio Return'
ax.set_xlabel(xAxLabel, fontproperties=axLblFont, color='k')
ax.set_ylabel(yAxLabel, fontproperties=axLblFont, color='k', labelpad=2)
ax.tick_params(axis='both', which='major', labelsize=8)
ax.set_xlim([0, 1.1*max(sigP)])
ax.set_ylim([0, 1.1*max(muP)])
fig.suptitle('Efficient Frontier of Portfolio', weight='bold', fontsize=10, x=0.4, y=0.95)
plt.show()
"""
