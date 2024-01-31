# simple_CNNs

Simple CNN blocks for easier re-use of small parts for when importing a full model isn't viable (anywhere people throw in a VGG or ResNet blindly). Originally I was going to do ResNet block -> DenseNet block -> ConvNext block to scale up design complexity and performance but the design of the ConvNext is basically as simple as the ResNet's if not even simpler, so I'll just focus on it.

ConvNext v1 and not v2.

I realize I could have just imported the keras.applications.convnext.py file and re-built juste the global wrapper/model building function, but at least now everything is clear and it's more independant from the rest of the system. Still needs to import keras to work but as long as the system/environment is capable of reproducing the 4-5 operations in there you could re-create it easily.

Differences: 
- No Layer Scale that's in the paper and the keras official implementation
- No Stochastic Depth that's in the paper and the keras official implementation
- No float16/other dtype optimisation (no info about it on the official?)
- No data/inputs check or verification that are in the official implementation
- No pre-trained weights obviously
- 

Sources/Refs: 
- [Official ConvNext Keras implementation](https://github.com/keras-team/keras/blob/v3.0.2/keras/applications/convnext.py)
- [ConvNext Paper](https://arxiv.org/abs/2201.03545)
