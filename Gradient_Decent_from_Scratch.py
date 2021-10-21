# Plan is to impliment gradient decent from scratch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Our data is based on housing prices. Kaggle (https://www.kaggle.com/c/house-prices-advanced-regression-techniques). It has shape: (1460, 81)
data = pd.read_csv('train.csv')

#pick indept and dept variables
x = data['GrLivArea']
y = data['SalePrice']

#normalise 
x_norm = (x - x.mean()) / x.std()

# add a column of ones for gradient decent
# np.c_ --> Translates slice objects to concatenation along the second axis (along column -- ie hstack).
x = np.c_[np.ones(x_norm.shape[0]), x_norm]


#Gradient Decent
# we need to impliment theta_{k+1} = theta_{k} - alpha * grad_{theta}(L(theta_{k}))
def gradient_descent(x, y, theta, iterations, alpha):
    past_losses = []
    past_thetas = [theta]
    for i in range(iterations):
        prediction = np.dot(x, theta)
        error = prediction - y 
        loss = 1/(2*m) * np.dot(error.T, error)
        past_losses.append(loss)
        theta = theta - (alpha * (1/m) * np.dot(x.T, error))
        past_thetas.append(theta)
        
    return past_thetas, past_losses

# Maximum Likihood estimate. 
def max_lik_estimate(X, y):
    """ X: N x D matrix of training inputs
        y: N x 1 vector of training targets/observations
        returns: maximum likelihood parameters (D x 1)
    """
    N, D = X.shape
    # linalg.solve method solves for x's in the equations of the type ax=b. (i.e beta* is what we are solving for)
    beta_ml = np.linalg.solve(X.T @ X, X.T @ y) 
    return beta_ml  

if __name__ == "__main__":

    alpha = 0.01 #Step size
    iterations = 2000 #No. of iterations
    m = y.size #No. of data points
    np.random.seed(123) #Set the seed
    theta = np.random.rand(2) #initalise theta

    # get optimal theta optimised by gradient decent
    past_thetas, past_losses = gradient_descent(x, y, theta, iterations, alpha)
    optimal_theta = past_thetas[-1]
    # True (analytical) optimal theta
    max_lik = max_lik_estimate(x,y)
    
    print("Gradient Descent: {:.2f}, {:.2f}".format(optimal_theta[0], optimal_theta[1]))
    print('Maximum Likeihood Estimate: {:.2f}, {:.2f}'.format(max_lik[0],max_lik[1]))
    print('We have found the True value! Garanteed since MSE is convex')

