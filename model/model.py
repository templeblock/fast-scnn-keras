from tensorflow import keras
from .layers import pyramid_pooling, bottleneck

__all__ = ["create_fast_scnn"]


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

    learning_to_down_sample = keras.layers.Conv2D(
        32, 3, 2, padding="same")(input_tensor)
    learning_to_down_sample = keras.layers.BatchNormalization()(
        learning_to_down_sample)
    learning_to_down_sample = keras.layers.Activation("relu")(
        learning_to_down_sample)

    learning_to_down_sample = keras.layers.SeparableConv2D(
        48, 3, 2, padding="same")(learning_to_down_sample)
    learning_to_down_sample = keras.layers.BatchNormalization()(
        learning_to_down_sample)
    learning_to_down_sample = keras.layers.Activation("relu")(
        learning_to_down_sample)

    learning_to_down_sample = keras.layers.SeparableConv2D(
        64, 3, 2, padding="same")(learning_to_down_sample)
    learning_to_down_sample = keras.layers.BatchNormalization()(
        learning_to_down_sample)
    learning_to_down_sample = keras.layers.Activation("relu")(
        learning_to_down_sample)

    skip_connection = learning_to_down_sample

    # Global feature extractor

    global_feature_extractor = bottleneck(learning_to_down_sample,
                                          64, 2, expansion_factor)
    global_feature_extractor = bottleneck(global_feature_extractor,
                                          64, 1, expansion_factor)
    global_feature_extractor = bottleneck(global_feature_extractor,
                                          64, 1, expansion_factor)

    global_feature_extractor = bottleneck(global_feature_extractor,
                                          96, 2, expansion_factor)
    global_feature_extractor = bottleneck(global_feature_extractor,
                                          96, 1, expansion_factor)
    global_feature_extractor = bottleneck(global_feature_extractor,
                                          96, 1, expansion_factor)

    global_feature_extractor = bottleneck(global_feature_extractor,
                                          128, 1, expansion_factor)
    global_feature_extractor = bottleneck(global_feature_extractor,
                                          128, 1, expansion_factor)
    global_feature_extractor = bottleneck(global_feature_extractor,
                                          128, 1, expansion_factor)
    global_feature_extractor = pyramid_pooling(global_feature_extractor,
                                               sub_region_sizes)

    # Feature fusion

    feature_fusion_main_branch = keras.layers.UpSampling2D((4, 4))(
        global_feature_extractor)

    feature_fusion_main_branch = keras.layers.DepthwiseConv2D(
        3, padding="same")(feature_fusion_main_branch)
    feature_fusion_main_branch = keras.layers.BatchNormalization()(
        feature_fusion_main_branch)
    feature_fusion_main_branch = keras.layers.Activation("relu")(
        feature_fusion_main_branch)
    feature_fusion_main_branch = keras.layers.Conv2D(
        128, 1, 1, padding="same")(feature_fusion_main_branch)
    feature_fusion_main_branch = keras.layers.BatchNormalization()(
        feature_fusion_main_branch)

    feature_fusion_skip_connection = keras.layers.Conv2D(
        128, 1, 1, padding="same")(skip_connection)
    feature_fusion_skip_connection = keras.layers.BatchNormalization()(
        feature_fusion_skip_connection)

    feature_fusion = feature_fusion_main_branch + feature_fusion_skip_connection

    # Classifier

    classifier = keras.layers.SeparableConv2D(128, 3, 1, padding="same")(
        feature_fusion)
    classifier = keras.layers.BatchNormalization()(classifier)
    classifier = keras.layers.Activation("relu")(classifier)

    classifier = keras.layers.SeparableConv2D(128, 3, 1, padding="same")(
        classifier)
    classifier = keras.layers.BatchNormalization()(classifier)
    classifier = keras.layers.Activation("relu")(classifier)

    classifier = keras.layers.Conv2D(num_classes, 3, 1, padding="same")(
        classifier)
    classifier = keras.layers.BatchNormalization()(classifier)
    classifier = keras.layers.Activation("relu")(classifier)

    output_tensor = keras.layers.UpSampling2D((8, 8))(classifier)
    output_tensor = keras.layers.Softmax()(output_tensor)

    model = keras.models.Model(input_tensor, output_tensor)
    return model
