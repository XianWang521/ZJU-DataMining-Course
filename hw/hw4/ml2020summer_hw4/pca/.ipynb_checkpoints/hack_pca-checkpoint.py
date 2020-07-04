import numpy as np
import matplotlib.pyplot as plt
from pca import PCA
from scipy import misc
def hack_pca(filename):
    '''
    Input: filename -- input image file name/path
    Output: img -- image without rotation
    '''
    img_r = (plt.imread(filename)).astype(np.float64)

    # YOUR CODE HERE
    # begin answer
    pixels =np.array(np.where(img_r[:, :, -1] > 0))
    vec, val = PCA(pixels.T)
    angle = np.arccos(vec[0,0]/np.linalg.norm(vec[:,0]))/np.pi*180 - 90;
    print(angle)
    return misc.imrotate(img_r, angle)
    
    # end answer