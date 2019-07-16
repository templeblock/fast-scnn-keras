from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import Input, Softmax
from layers import PyramidPoolingModule, Bottleneck


def create_fast_scnn(num_classes, input_shape=[1024, 2048, 3], 
                     sub_region_sizes=[1, 2, 3, 6], expansion_factor=6):
    """This function creates a Fast-SCNN neural network model using 
    the Keras functional API.
    
    Args:
        num_classes: Number of classes
        input_shape: A list containing information about the size of the image.
            List structure: (rows, cols, channels). Dimensions can also be
            None if they can be of any size.
        expansion_factor: Hyperparameter in the bottleneck layer
        sub_region_sizes: A list containing the sizes of subregions for
            average pool by region in the pyramidal pool module
    
    Returns:
        model: uncompiled Keras model
    """

    # Sub-models for every Fast-SCNN block

    input_tensor = Input(input_shape)
    
    learning_to_down_sample = Sequential([
        Conv2D(32, 3, 2,padding="same"),
        BatchNormalization(),
        Activation("relu"),

        SeparableConv2D(48, 3, 2, padding="same"),
        BatchNormalization(),
        Activation("relu"),

        SeparableConv2D(48, 3, 2, padding="same"),
        BatchNormalization(),
        Activation("relu"),
    ])

    global_feature_extractor = Sequential([
        Bottleneck(64, 2, expansion_factor),
        Bottleneck(64, 1, expansion_factor),
        Bottleneck(64, 1, expansion_factor),

        Bottleneck(96, 2, expansion_factor),
        Bottleneck(96, 1, expansion_factor),
        Bottleneck(96, 1, expansion_factor),

        Bottleneck(128, 1, expansion_factor),
        Bottleneck(128, 1, expansion_factor),
        Bottleneck(128, 1, expansion_factor),
    ])

    feature_fusion_main_branch = Sequential([
        UpSampling2D((4, 4)),
        DepthwiseConv2D(3, padding="same"),
        BatchNormalization(),
        Activation("relu"),

        Conv2D(128, 1, 1, padding="same"),
        BatchNormalization()
    ])

    feature_fuison_skip_connection = Sequential([
        Conv2D(128, 1, 1, padding="same"),
        BatchNormalization()
    ])

    classifier = Sequential([
        SeparableConv2D(128, 3, 1, padding="same"),
        BatchNormalization(),
        Activation("relu"),

        SeparableConv2D(128, 3, 1, padding="same"),
        BatchNormalization(),
        Activation("relu"),

        Conv2D(num_classes, 3, 1, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        
        UpSampling2D((8, 8)),
        Softmax()
    ])

    # Using sub-models on input

    tensor = learning_to_down_sample(input_tensor)
    skip_connection = tensor
    tensor = global_feature_extractor(tensor)
    tensor = PyramidPoolingModule(sub_region_sizes)(tensor)
    tensor = feature_fusion_main_branch(tensor) +\
            feature_fuison_skip_connection(skip_connection)
    output_tensor = classifier(tensor)
    model = Model(input_tensor, output_tensor)
    return model
