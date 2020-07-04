import numpy as np

def perceptron(X, y):
    '''
    PERCEPTRON Perceptron Learning Algorithm.

       INPUT:  X: training sample features, P-by-N matrix.
               y: training sample labels, 1-by-N row vector.

       OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
               iter: number of iterations

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    iters = 0
    # YOUR CODE HERE
    
    # begin answer
    MAX_ITERS = 1e4
    while(iters < MAX_ITERS):
        flag = True
        for i in range(N):
            iters += 1
            x_i = np.hstack((np.ones((1)), X[:,i]))
            x_i = x_i.reshape(x_i.shape[0], 1) 
            if np.sign(np.matmul(w.T, x_i))[0][0] != y[0,i]:
                flag = False
                w += x_i * y[0,i]
        if flag:
            break
    # end answer
    
    return w, iters