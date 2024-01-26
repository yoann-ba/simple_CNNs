




#%% imports

import numpy as np
import matplotlib.pyplot as plt

from keras import layers
from keras import ops



#%% convnext

# 
def ConvNextBlock():
    
    # depthwise conv, 7x7, 96 channels
    # Layer Norm
    # 1x1 conv / Dense layer, 384 channels (x4)
    # GELU (/ReLU)
    # 1x1 conv / Dense layer, 96 channels
    # add inputs
    
    return


# 
# ConvNext is always 4 stages of img resolution
# nb of blocks per stages is either (3, 3, 9, 3) or (3, 3, 27, 3)
# nb of channels in the blocks goes x2 every stage
# beginning is a patchify stem?
def ConvNext():
    
    # stem
    
    # other preprocess?
    
    # for each stage
    # for each block
    
    # output/postprocess
    
    return







































































































