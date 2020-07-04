import numpy as np
import scipy.stats
from scipy.spatial.distance import cdist

def knn(x, x_train, y_train, k):
    '''
    KNN k-Nearest Neighbors Algorithm.

        INPUT:  x:         testing sample features, (N_test, P) matrix.
                x_train:   training sample features, (N, P) matrix.
                y_train:   training sample labels, (N, ) column vector.
                k:         the k in k-Nearest Neighbors

        OUTPUT: y    : predicted labels, (N_test, ) column vector.
    '''

    # Warning: uint8 matrix multiply uint8 matrix may cause overflow, take care
    # Hint: You may find numpy.argsort & scipy.stats.mode helpful

    # YOUR CODE HERE

    # begin answer
    dists = cdist(x_train, x)
    #refer from https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html
    index = np.argpartition(dists, k, axis = 0)[:k]
    
    neighbors = np.zeros(index.shape)
    for i in range(index.shape[0]):
        for j in range(index.shape[1]):
            neighbors[i][j] = y_train[index[i][j]]
    # refer from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html
    y = scipy.stats.mode(neighbors, axis = 0)[0]
    # end answer

    return y
