import numpy as np

def relu_backprop(in_sensitivity, in_):
    '''
    The backpropagation process of relu
      input paramter:
          in_sensitivity  : the sensitivity from the upper layer, shape: 
                          : [number of images, number of outputs in feedforward]
          in_             : the input in feedforward process, shape: same as in_sensitivity
      
      output paramter:
          out_sensitivity : the sensitivity to the lower layer, shape: same as in_sensitivity
    '''
    # TODO

    # begin answer
    out_sensitivity = in_sensitivity.copy()
    for i in range(in_.shape[0]):
        for j in range(in_.shape[1]):
            if(in_[i][j] < 0):
                out_sensitivity[i][j] = 0
    # end answer
    return out_sensitivity
