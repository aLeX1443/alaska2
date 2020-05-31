import efficientnet.tfkeras as efn
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


def build_efficientnet_model(input_shape, print_model_summary=True):
    efficientnet_model = efn.EfficientNetB7(
        weights="imagenet",  # 'noisy-student',
        input_shape=input_shape,
        include_top=False,
        pooling="avg",
    )

    # Define the input layer.
    cnn_input = Input(shape=input_shape, name="cnn_input")

    # Create a feature vector using the pre-trained EfficientNet model.
    feature_vector = efficientnet_model(cnn_input)

    # Add MLP block to generate a prediction.
    x = Dense(1024, activation="relu")(feature_vector)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=[cnn_input], outputs=[output])

    if print_model_summary:
        print(model.summary())

    return model


def build_base_efficientnet_model(input_shape, print_model_summary=True):
    model = Sequential(
        [
            efn.EfficientNetB3(
                input_shape=(512, 512, 3),
                weights="imagenet",
                include_top=False,
            ),
            GlobalAveragePooling2D(),
            Dense(1, activation="sigmoid", dtype="float32"),
        ]
    )

    if print_model_summary:
        print(model.summary())

    return model
