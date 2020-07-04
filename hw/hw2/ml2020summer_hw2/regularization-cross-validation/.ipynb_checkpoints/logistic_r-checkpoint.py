import numpy as np

def logistic_r(X, y, lmbda):
    '''
    LR Logistic Regression.

      INPUT:  X:   training sample features, P-by-N matrix.
              y:   training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    iter = 0
    x_i = np.vstack((np.ones((1, X.shape[1])), X))
    theta = 0.01
    learn_rate = 0.001
    while(iter < 1000):
        grad = np.matmul(-(y-1/(1+np.exp(-np.matmul(w.T, x_i)))), x_i.T).T + 2*lmbda*np.abs(w)
        if(np.linalg.norm(grad) < theta):
            break
        w -= grad * learn_rate
        iter += 1
    # end answer
    return w
