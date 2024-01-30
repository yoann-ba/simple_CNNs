




#%% imports

import keras
from keras import layers



#%% convnext

# 
# Small -> Large -> Small pattern
#TODO add stage and block nb to name
def ConvNextBlock(x, nb_c, model_name):
    
    # depthwise grouped conv, 7x7, 96 channels
    x2 = layers.Conv2D(filters = nb_c, 
                       kernel_size = 7, 
                       padding = "same", 
                       groups = nb_c, 
                       name = model_name + "_depthwise_conv")(x)
    # Layer Norm
    x2 = layers.LayerNormalization(epsilon=1e-6, name = model_name + "_layernorm")(x2)
    # 1x1 conv / Dense layer, 384 channels (x4)
    x2 = layers.Dense(4 * nb_c, name = model_name + "_pointwise_conv_1")(x2)
    # GELU (/ReLU)
    x2 = layers.Activation("gelu", name = model_name + "_gelu")(x2)
    # 1x1 conv / Dense layer, 96 channels
    x2 = layers.Dense(nb_c, name = model_name + "_pointwise_conv_2")(x2)
    
    # bind the skip connection
    x = x + x2
    
    # Layer Scale?
    # Stochastic Depth
    
    return x

# 
# reduces intial input data by 4
def Stem(x, nb_c, model_name):
    
    x = layers.Conv2D(nb_c, 
                      kernel_size = 4, strides = (4, 4), 
                      name = model_name + '_stem_conv')(x)
    x = layers.LayerNormalization(epsilon = 1e-6, 
                                  name = model_name + '_stem_layernorm')(x)
    
    return x


# 
#TODO add stage nb to name
# if we don't want to downsample then it comes out same size but half reduced in details
def Downsampling(x, nb_c, model_name, downsample = True):
    
    x = layers.LayerNormalization(epsilon = 1e-6, 
                                  name = model_name + '_downsampling_layernorm')(x)
    if downsample:
        x = layers.Conv2D(filters = nb_c, 
                      kernel_size = 2, 
                      strides = 2, 
                      name=model_name + '_downsampling_conv')(x)
    else:
        x = layers.MaxPooling2D(padding='same', strides = (1, 1), 
                                name = model_name + '_same_size_maxpool')(x)
        # x = layers.Conv2D(filters = nb_c, 
        #                   kernel_size = 2, 
        #                   name = model_name + '_same_size_conv')(x)
    # if
    
    return x

# 
#TODO parametrize classif activation function?
def ClassifierHead(x, nb_classes, model_name):
    
    x = layers.GlobalAveragePooling2D(name=model_name + "_head_gap")(x)
    x = layers.LayerNormalization(epsilon=1e-6, 
                                  name=model_name + "_head_layernorm")(x)
    x = layers.Dense(nb_classes, activation="softmax", 
                     name=model_name + "_head_dense")(x)
    
    return x

# 
# ConvNext is always 4 stages of img resolution
# nb of blocks per stages is either (3, 3, 9, 3) or (3, 3, 27, 3)
# nb of channels in the blocks goes x2 every stage
# beginning is a patchify stem? 4x4 non overlap convolution

# just deduce the number of stages from the length of nb blocks
#TODO stochastic depth
#TODO layer scale
#TODO float16?
def ConvNext(nb_blocks = (3, 3, 9, 3), 
             nb_channels = (96, 192, 384, 768), 
             input_shape = None, classif_head = False, nb_classes = None, 
             downsample = True, 
             model_name = 'homemade_convnext'):
    
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # pre stem (standard scaling according to some mu and sigma)
    # just do it on your data before feeding it here
    # or declare another model that goes preproc->this model
    
    # stem
    x = Stem(x, nb_c=nb_channels[0], model_name=model_name)
    
    # for each stage
    for i_s in range(len(nb_blocks)):
        # for each block
        for i_b in range(nb_blocks[i_s]):
            x = ConvNextBlock(x, nb_c=nb_channels[i_s], 
                              model_name=f"{model_name}_s{i_s+1}_b{i_b+1}")
        if i_s < len(nb_blocks)-1: # no downsample on the last one
            x = Downsampling(x, nb_c=nb_channels[i_s + 1], 
                             model_name=f"{model_name}_s{i_s+1}_b{i_b+1}")
    # end stage loop
    
    # output/postprocess
    if classif_head == True:
        if nb_classes != None:
            x = ClassifierHead(x, nb_classes, model_name=model_name)
    # 
    
    model = keras.Model(inputs = inputs, outputs = x, name = model_name)
    
    return model

# Training:
# AdamW opt
# 300 epochs
# data aug: MixUp, CutMix, RandAugment, Random Erasing
# regularization: Stochastic Depth, Label Smoothing
# full settings of hyperparameters for pre-training and fine-tuning available in Appendix



































































































