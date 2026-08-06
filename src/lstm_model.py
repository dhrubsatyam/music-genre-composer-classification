"""LSTM branch — operates on the integer-encoded note sequence (pitch, dur, ioi).

Report Section 6.2 shows a single Embedding(input_dim=130) feeding stacked
Bidirectional LSTMs. Our sequences carry three integer channels, so we embed
each channel and concatenate — a strict superset of the report design that
uses the duration/offset information as well as pitch.

Masking: the note-sequence arrays must be pre-remapped so the pad token is 0
in every channel (see train.remap_pad_to_zero). With mask_zero=True on each
embedding, padded time-steps are masked out of both LSTM directions.
"""
import keras
from keras import layers


def build_lstm(seq_len=128, pitch_vocab=129, dur_vocab=13, ioi_vocab=14,
               n_classes=4, pitch_emb=100, dur_emb=8, ioi_emb=8,
               lstm1=256, lstm2=128, dropout=0.3, rec_dropout=0.2,
               dense_units=64, lr=1e-3):
    """Return a compiled Bidirectional-LSTM model for the sequence branch.

    Expects input of shape (seq_len, 3) with pad already remapped to 0.
    """
    inp = layers.Input(shape=(seq_len, 3), dtype="int32", name="note_sequence")

    pitch = layers.Lambda(lambda z: z[:, :, 0], name="pitch")(inp)
    dur   = layers.Lambda(lambda z: z[:, :, 1], name="dur")(inp)
    ioi   = layers.Lambda(lambda z: z[:, :, 2], name="ioi")(inp)

    ep = layers.Embedding(pitch_vocab, pitch_emb, mask_zero=True, name="emb_pitch")(pitch)
    ed = layers.Embedding(dur_vocab,   dur_emb,   mask_zero=True, name="emb_dur")(dur)
    ei = layers.Embedding(ioi_vocab,   ioi_emb,   mask_zero=True, name="emb_ioi")(ioi)
    x = layers.Concatenate(name="emb_concat")([ep, ed, ei])

    x = layers.Bidirectional(
        layers.LSTM(lstm1, return_sequences=True, recurrent_dropout=rec_dropout),
        name="bilstm1")(x)
    x = layers.Dropout(dropout, name="dropout1")(x)
    x = layers.Bidirectional(
        layers.LSTM(lstm2, recurrent_dropout=rec_dropout),
        name="bilstm2")(x)
    x = layers.Dropout(dropout, name="dropout2")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense")(x)
    out = layers.Dense(n_classes, activation="softmax", name="composer")(x)

    model = keras.Model(inp, out, name="lstm_sequence")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model
