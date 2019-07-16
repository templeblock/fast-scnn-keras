import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D
from tensorflow.keras.layers import AveragePooling2D, Layer
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import Lambda, Concatenate


class PyramidPoolingModule(Layer):
    """This class implements the Pyramid Pooling Module
    
    WARNING: This class uses eager execution, so it only works with
        Tensorflow 2.0 backend.
    
    Arguments
        sub_region_size: A list containing the size of each region for the
            sub-region average pooling. The default value is [1, 2, 3, 6]
            
    Input shape
        Tensor with shape: (batch, rows, cols, channels)
        
    Output shape
        Tensor with shape: (batch, rows, cols, channels * 2)
    """
    def __init__(self, sub_region_sizes, **kwargs):
        self.sub_region_sizes = sub_region_sizes
        super(PyramidPoolingModule, self).__init__(dynamic=True, **kwargs)

    def call(self, tensor):
        _, input_height, input_width, input_channels = tensor.shape
        feature_maps = [tensor]
        for i in self.sub_region_sizes:
            curr_feature_map = AveragePooling2D(
                pool_size=(input_height // i, input_width // i),
                strides=(input_height // i, input_width // i))(tensor)
            curr_feature_map = Conv2D(
                    filters=int(input_channels) // len(self.sub_region_sizes),
                    kernel_size=3,
                    padding='same')(curr_feature_map)
            curr_feature_map = Lambda(
                lambda x: tf.image.resize(
                    x, (input_height, input_width)))(curr_feature_map)
            feature_maps.append(curr_feature_map)

        output_tensor = Concatenate(axis=-1)(feature_maps)
        
        output_tensor = Conv2D(filters=128, kernel_size=3, strides=1,
                                    padding="same")(output_tensor)
        output_tensor = BatchNormalization()(output_tensor)
        output_tensor = Activation("relu")(output_tensor)
        return output_tensor

    def compute_output_shape(self, input_shape):
        return input_shape
    

class Bottleneck(Layer):
    """Implementing Bottleneck.
    
    This class implements the bottleneck module for Fast-SCNN.
    Layer structure:
        ----------------------------------------------------------------
        |  Input shape   |  Block  |  Kernel | Stride |  Output shape  |
        |                |         |   size  |        |                |
        |----------------|---------|---------|--------|----------------|
        |   h * w * c    |  Conv2D |    1    |    1   |   h * w * tc   |
        |----------------|---------|---------|--------|----------------|
        |   h * w * tc   |  DWConv |    3    |    s   | h/s * w/s * tc |
        |----------------|---------|---------|--------|----------------|
        | h/s * w/s * tc |  Conv2D |    1    |    1   | h/s * w/s * c` |
        |--------------------------------------------------------------|
    
        Designations:
            h: input height
            w: input width
            c: number of input channels
            t: expansion factor
            c`: number of output channels
            DWConv: depthwise convolution

    # Arguments
        filters: Output filters
        strides: Stride used in depthwise convolution layer
        expansion_factor: hyperparameter
        
    # Input shape
        Tensor with shape: (batch, rows, cols, channels)
        
    # Output shape
        Tensor with shape: (batch, rows // stride, cols // stride,
                            new_channels)
    """
    def __init__(self, filters, strides, expansion_factor, **kwargs):
        self.filters = filters
        self.strides = strides
        self.expansion_factor = expansion_factor
        super(Bottleneck, self).__init__(**kwargs)
    
    def call(self, tensor):
        _, input_height, input_width, input_channels = tensor.shape
        tensor = Conv2D(filters=input_channels * self.expansion_factor,
                        kernel_size=1,
                        strides=1,
                        padding="same",
                        activation="relu")(tensor)
        tensor = BatchNormalization()(tensor)
        tensor = Activation('relu')(tensor)

        tensor = DepthwiseConv2D(kernel_size=3,
                                 strides=self.strides,
                                 padding="same")(tensor)
        tensor = BatchNormalization()(tensor)
        tensor = Activation('relu')(tensor)

        tensor = Conv2D(filters=self.filters,
                        kernel_size=1,
                        strides=1,
                        padding="same")(tensor)
        tensor = BatchNormalization()(tensor)
        tensor = Activation('relu')(tensor)

        return tensor