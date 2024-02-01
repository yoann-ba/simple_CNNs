




#%% imports

import numpy as np

import keras
from keras import layers

# for custom layers
from keras.layers.layer import Layer
from keras import ops, random # for stochastic depth
from keras import initializers # for layer scale


#%% Custom layers
# taken blindly from 
# https://github.com/keras-team/keras/blob/v3.0.2/keras/applications/convnext.py


class StochasticDepth(Layer):
    """Stochastic Depth module.

    It performs batch-wise dropping rather than sample-wise. In libraries like
    `timm`, it's similar to `DropPath` layers that drops residual paths
    sample-wise.

    References:
    - https://github.com/rwightman/pytorch-image-models

    Args:
      drop_path_rate (float): Probability of dropping paths. Should be within
        [0, 1].

    Returns:
      Tensor either with the residual path dropped or kept.
    """

    def __init__(self, drop_path_rate, **kwargs):
        super().__init__(**kwargs)
        self.drop_path_rate = drop_path_rate

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path_rate
            shape = (ops.shape(x)[0],) + (1,) * (len(ops.shape(x)) - 1)
            random_tensor = keep_prob + random.uniform(shape, 0, 1)
            random_tensor = ops.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_path_rate": self.drop_path_rate})
        return config


# essentially x <- x * gamma (~1e-6)
class LayerScale(Layer):
    """Layer scale module.

    References:

    - https://arxiv.org/abs/2103.17239

    Args:
        init_values (float): Initial value for layer scale. Should be within
            [0, 1].
        projection_dim (int): Projection dimensionality.

    Returns:
        Tensor multiplied to the scale.
    """

    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, _):
        self.gamma = self.add_weight(
            shape=(self.projection_dim,),
            initializer=initializers.Constant(self.init_values),
            trainable=True,
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config



#%% convnext

# 
# Small -> Large -> Small pattern
def ConvNextBlock(x, nb_c, model_name, 
                  layer_scale_init_value = 1e-6, drop_path_rate = 0.0):
    
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
    
    # Layer Scale, i.e x = x * gamma
    if layer_scale_init_value is not None:
        x = LayerScale(layer_scale_init_value,
                       nb_c, name = model_name + "_layer_scale")(x)
    # Stochastic Depth
    if drop_path_rate:
        x = StochasticDepth(drop_path_rate, 
                            name = model_name + "_stochastic_depth")(x)
    else:
        x = layers.Activation("linear", 
                              name = model_name + "_identity")(x)
    # --
    
    # bind the skip connection
    x = x + x2
    
    
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
#TODO float16?
def ConvNext(nb_blocks = (3, 3, 9, 3), 
             nb_channels = (96, 192, 384, 768), 
             input_shape = None, classif_head = False, nb_classes = None, 
             downsample = True, 
             layer_scale_init_value = 1e-6, drop_path_rate = 0.0, 
             model_name = 'homemade_convnext'):
    
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # stem
    x = Stem(x, nb_c=nb_channels[0], model_name=model_name)
    
    # Block 0 get 0% of the drop rate, 
    # Block n gets 100% of input drop rate, linear in between
    depth_drop_rates = [float(x) for x in np.linspace(0.0, drop_path_rate, sum(nb_blocks))]
    
    block_counter = 0
    # for each stage
    for i_s in range(len(nb_blocks)):
        # for each block
        for i_b in range(nb_blocks[i_s]):
            x = ConvNextBlock(x, nb_c=nb_channels[i_s], 
                              layer_scale_init_value=layer_scale_init_value, 
                              drop_path_rate=depth_drop_rates[block_counter], 
                              model_name=f"{model_name}_s{i_s+1}_b{i_b+1}")
            block_counter += 1
        if i_s < len(nb_blocks)-1: # no downsample on the last one
            x = Downsampling(x, nb_c=nb_channels[i_s + 1], 
                             model_name=f"{model_name}_s{i_s+1}_b{i_b+1}")
            # 
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



































































































