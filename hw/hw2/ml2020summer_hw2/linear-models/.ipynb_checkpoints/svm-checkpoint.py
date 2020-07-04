import numpy as np
import scipy.optimize as opt
def func(w, X, y):
    return np.multiply(y[0, :], np.matmul(w.T, X)) -1
def f(w):
    return 0.5*np.linalg.norm(w[1:,]) * np.linalg.norm(w[1:,])

def svm(X, y):
    '''
    SVM Support vector machine.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
            num: number of support vectors

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    num = 0

    # YOUR CODE HERE
    # Please implement SVM with scipy.optimize. You should be able to implement
    # it within 20 lines of code. The optimization should converge wtih any method
    # that support constrain.
    # begin answer
    # refer from https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize 
    x_i = np.vstack((np.ones((1,N)), X))
    con = {'type': 'ineq', 'fun': func, 'args': (x_i, y)}
    res = opt.minimize(f, w, constraints=con, method='SLSQP')
    w = res.x
    w_y = np.matmul(x_i.T, w).T
    num = N - np.sum(y*w_y < 0.9) - np.sum(y*w_y > 1.1)
    # end answer
    return w, num

