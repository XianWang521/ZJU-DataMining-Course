import numpy as np

def logistic(X, y):
    '''
    LR Logistic Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    iter = 0
    x_i = np.vstack((np.ones((1, X.shape[1])), X))
    theta = 0.1
    learn_rate = 0.01
    while(iter < 1000):
        #the calculation of grad is from P60 in book《Machine Learning》 written by Zhihua Zhou
        grad = np.matmul(-(y-1/(1+np.exp(-np.matmul(w.T, x_i)))), x_i.T).T
        if(np.linalg.norm(grad) < theta):
            break
        w -= grad * learn_rate
        iter += 1
    # end answer
    return w
