"""CNN branch — operates on the 128 x T velocity-weighted piano-roll.

Layer design follows report Section 6.1 (Conv32 -> Pool -> Conv64 -> Pool ->
Conv128 -> GlobalAveragePooling -> Dense128 -> Softmax). The stack is
time-length agnostic thanks to GlobalAveragePooling2D, so it accepts our real
128 x 128 excerpts (the report's "500" was an illustrative T).
"""
import keras
from keras import layers


def build_cnn(input_shape=(128, 128, 1), n_classes=4,
              filters=(32, 64, 128), dense_units=128,
              dropout=0.4, l2=1e-5, lr=1e-3):
    """Return a compiled CNN model for the piano-roll branch."""
    reg = keras.regularizers.l2(l2) if l2 else None
    inp = layers.Input(shape=input_shape, name="piano_roll")

    x = inp
    # Two conv blocks with pooling, one conv block feeding global pooling.
    for i, f in enumerate(filters):
        x = layers.Conv2D(f, 3, padding="same", kernel_regularizer=reg,
                          name=f"conv{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn{i+1}")(x)
        x = layers.Activation("relu", name=f"relu{i+1}")(x)
        if i < len(filters) - 1:                      # pool after all but last
            x = layers.MaxPooling2D(2, name=f"pool{i+1}")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(dense_units, activation="relu",
                     kernel_regularizer=reg, name="dense")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(n_classes, activation="softmax", name="composer")(x)

    model = keras.Model(inp, out, name="cnn_pianoroll")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model
