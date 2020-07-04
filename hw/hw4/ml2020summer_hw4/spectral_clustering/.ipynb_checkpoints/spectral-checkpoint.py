import numpy as np
from kmeans import kmeans

def spectral(W, k):
    '''
    SPECTRUAL spectral clustering

        Input:
            W: Adjacency matrix, N-by-N matrix
            k: number of clusters

        Output:
            idx: data point cluster labels, n-by-1 vector.
    '''
    # YOUR CODE HERE
    # begin answer
    n = W.shape[0]
    D = np.diag(np.sum(W, axis = 0))
    L = D - W
    #calculate eigenvalues and eigenvectors
    val, vec = np.linalg.eigh(L)
    idx = kmeans(vec[:, :k], k)
    return idx
    # end answer
