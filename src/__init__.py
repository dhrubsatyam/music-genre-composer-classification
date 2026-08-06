"""Model, training, and evaluation helpers for composer classification.

Modules
-------
cnn_model : build_cnn   — CNN over the piano-roll representation
lstm_model: build_lstm  — Bidirectional-LSTM over the note-sequence representation
train     : grouped splitting, class weights, pitch-transposition augmentation, callbacks
evaluate  : metrics, confusion matrices, late-fusion hybrid + fusion-weight tuning
"""
