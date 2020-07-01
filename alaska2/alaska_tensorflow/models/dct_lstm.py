import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    LSTM,
    Bidirectional,
)


def build_dct_lstm(input_shape, n_classes=4, print_model_summary=True):
    recurrent_units = 64
    dropout_rate = 0.0
    n_blocks = 5  # 10
    dense_units = 192

    numerical_input = tf.keras.layers.Input(
        shape=input_shape, name="numerical_input"
    )

    x = Bidirectional(LSTM(recurrent_units, return_sequences=True))(
        numerical_input
    )
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)

    for _ in range(n_blocks):
        x = Bidirectional(LSTM(recurrent_units, return_sequences=True))(x)
        x = Dropout(dropout_rate)(x)
        x = BatchNormalization()(x)

    x = Bidirectional(LSTM(recurrent_units, return_sequences=False))(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)

    x = Dense(dense_units, activation="relu")(x)
    output = Dense(n_classes, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs=[numerical_input], outputs=[output])

    if print_model_summary:
        print(model.summary())

    return model


if __name__ == "__main__":
    model = build_dct_lstm(input_shape=(12288, 64))
