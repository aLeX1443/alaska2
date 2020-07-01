import efficientnet.tfkeras as efn
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


def build_efficientnet_b3(input_shape, n_classes=4, print_model_summary=True):
    efficientnet_model = efn.EfficientNetB3(
        weights="imagenet",  # 'noisy-student',
        input_shape=input_shape,
        classes=n_classes,
        pooling="avg",
        include_top=False,
    )

    # Define the input layer.
    cnn_input = Input(shape=input_shape, name="cnn_input")

    # Create a feature vector using the pre-trained EfficientNet model.
    x = efficientnet_model(cnn_input)

    # Add MLP block to generate a prediction.
    x = Dense(1024, activation="relu")(x)
    output = Dense(
        n_classes, activation="sigmoid" if n_classes == 1 else "softmax"
    )(x)

    model = Model(inputs=[cnn_input], outputs=[output])

    if print_model_summary:
        print(model.summary())

    return model


def build_base_efficientnet_model(
    input_shape, n_classes=4, print_model_summary=True
):
    model = Sequential(
        [
            efn.EfficientNetB3(
                input_shape=(3, 512, 512),
                weights="imagenet",
                include_top=False,
            ),
            GlobalAveragePooling2D(),
            Dense(n_classes, activation="softmax", dtype="float32"),
        ]
    )

    if print_model_summary:
        print(model.summary())

    return model
