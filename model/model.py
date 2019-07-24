from tensorflow import keras
from model.layers import PyramidPoolingModule, Bottleneck


def create_fast_scnn(num_classes, input_shape=[None, None, 3],
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

    input_tensor = keras.layers.Input(input_shape)

    learning_to_down_sample = keras.models.Sequential([
        keras.layers.Conv2D(32, 3, 2, padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),

        keras.layers.SeparableConv2D(48, 3, 2, padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),

        keras.layers.SeparableConv2D(48, 3, 2, padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
    ])

    global_feature_extractor = keras.models.Sequential([
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

    feature_fusion_main_branch = keras.models.Sequential([
        keras.layers.UpSampling2D((4, 4)),
        keras.layers.DepthwiseConv2D(3, padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),

        keras.layers.Conv2D(128, 1, 1, padding="same"),
        keras.layers.BatchNormalization()
    ])

    feature_fusion_skip_connection = keras.models.Sequential([
        keras.layers.Conv2D(128, 1, 1, padding="same"),
        keras.layers.BatchNormalization()
    ])

    classifier = keras.models.Sequential([
        keras.layers.SeparableConv2D(128, 3, 1, padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),

        keras.layers.SeparableConv2D(128, 3, 1, padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),

        keras.layers.Conv2D(num_classes, 3, 1, padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),

        keras.layers.UpSampling2D((8, 8)),
        keras.layers.Softmax()
    ])

    # Using sub-models on input

    tensor = learning_to_down_sample(input_tensor)
    skip_connection = tensor
    tensor = global_feature_extractor(tensor)
    tensor = PyramidPoolingModule(sub_region_sizes)(tensor)
    tensor = feature_fusion_main_branch(
        tensor) + feature_fusion_skip_connection(skip_connection)
    output_tensor = classifier(tensor)
    model = keras.models.Model(input_tensor, output_tensor)
    return model


def loss_function(y_true, y_pred):
    loss_value = keras.backend.categorical_crossentropy(y_true, y_pred)
    loss_value = keras.backend.sum(loss_value, axis=[1, 2])
    loss_value = keras.backend.mean(loss_value)
    return loss_value


def custom_accuracy_metric(y_true, y_pred):
    y_true = keras.backend.argmax(y_true)
    y_pred = keras.backend.argmax(y_pred)
    correct_predictions = keras.backend.sum(keras.backend.ones_like(
        y_true.shape, "float32"))
    all_predictions = keras.backend.sum(keras.backend.cast(
        keras.backend.equal(y_true, y_pred), "float32"), axis=[1, 2])
    accuracy = correct_predictions / all_predictions
    accuracy = keras.backend.mean(accuracy)
    return accuracy
