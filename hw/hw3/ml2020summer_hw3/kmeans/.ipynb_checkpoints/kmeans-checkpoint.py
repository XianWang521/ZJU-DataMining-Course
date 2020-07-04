import numpy as np

def kmeans(x, k):
    '''
    KMEANS K-Means clustering algorithm

        Input:  x - data point features, n-by-p maxtirx.
                k - the number of clusters

        OUTPUT: idx  - cluster label
                ctrs - cluster centers, K-by-p matrix.
                iter_ctrs - cluster centers of each iteration, (iter, k, p)
                        3D matrix.
    '''
    # YOUR CODE HERE

    # begin answer
    MAX_ITER = 1000
    # Randomly assign category center position
    idx = np.random.randint(0, k, (x.shape[0], ))
    ctrs = x[np.random.randint(x.shape[0], size = k), :]
    iter_ctrs = np.zeros((MAX_ITER+1, k, x.shape[1]))
    iter_ctrs[0, :, :] = ctrs
    
    for i in range(1, MAX_ITER + 1):
        error = 0
        for j in range(x.shape[0]):
            dist = [np.sum(np.square(ctrs[m, :] - x[j, :])) for m in range(k)]
            if idx[j] != np.argmin(dist):
                error += 1
                idx[j] = np.argmin(dist)
        
        if error == 0:break
        
        for m in range(k):
            ctrs[m] = np.zeros((x.shape[1], ))
            index = np.array(idx == m)
            if index.sum() != 0:
                ctrs[m] = np.mean(x[index, :], axis = 0)
        
        iter_ctrs[i, :, :] = ctrs
    # end answer
    
    iter_ctrs.resize((i, k, x.shape[1]))
    return idx, ctrs, iter_ctrs