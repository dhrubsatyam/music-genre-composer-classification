"""Training utilities: leakage-safe splitting, class weights, augmentation, callbacks."""
import numpy as np
import keras
from sklearn.model_selection import GroupShuffleSplit


# --------------------------------------------------------------------------
# Leakage-safe train / val / test split (grouped by source file)
# --------------------------------------------------------------------------
def grouped_split(y, groups, test_size=0.15, val_size=0.15, seed=42):
    """Split indices so that all excerpts of a file stay in one fold.

    Returns (train_idx, val_idx, test_idx). val_size is a fraction of the
    whole dataset (re-derived relative to the train+val remainder).
    """
    idx = np.arange(len(y))
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval, test = next(gss.split(idx, y, groups))

    rel_val = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
    tr, va = next(gss2.split(trainval, y[trainval], groups[trainval]))
    train_idx, val_idx = trainval[tr], trainval[va]

    # sanity: no group appears in more than one split
    gt, gv, gs = set(groups[train_idx]), set(groups[val_idx]), set(groups[test])
    assert not (gt & gv) and not (gt & gs) and not (gv & gs), "group leakage!"
    return train_idx, val_idx, test


# --------------------------------------------------------------------------
# Inverse-frequency class weights (report Section 6.4)
# --------------------------------------------------------------------------
def class_weights(y_train, n_classes):
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    w = len(y_train) / (n_classes * counts)
    return {i: float(w[i]) for i in range(n_classes)}


# --------------------------------------------------------------------------
# Pitch-transposition data augmentation (report Section 6.4, +/- 2 semitones)
# --------------------------------------------------------------------------
def transpose_pianoroll(X, k):
    """Shift the piano roll up/down k semitones along the pitch axis (zero-fill).
    X: (N, 128, T)."""
    if k == 0:
        return X
    out = np.zeros_like(X)
    if k > 0:
        out[:, k:, :] = X[:, :128 - k, :]
    else:
        out[:, :128 + k, :] = X[:, -k:, :]
    return out


def transpose_sequence(S, k, pitch_pad=128):
    """Shift real pitches by k semitones (clip to 0..127), leaving pad untouched.
    S: (N, L, 3) integer array in the ORIGINAL (pre-remap) encoding."""
    if k == 0:
        return S
    out = S.copy()
    p = out[..., 0]
    mask = p != pitch_pad
    shifted = np.clip(p + k, 0, 127)
    p[mask] = shifted[mask]
    out[..., 0] = p
    return out


def augment_train(X_pr, X_seq, y, offsets=(-2, -1, 1, 2), pitch_pad=128):
    """Expand the training set with transposed copies. Returns stacked arrays
    including the originals (offset 0)."""
    prs, seqs, ys = [X_pr], [X_seq], [y]
    for k in offsets:
        prs.append(transpose_pianoroll(X_pr, k))
        seqs.append(transpose_sequence(X_seq, k, pitch_pad))
        ys.append(y)
    return (np.concatenate(prs), np.concatenate(seqs), np.concatenate(ys))


# --------------------------------------------------------------------------
# Sequence pad -> 0 remap (so Embedding(mask_zero=True) masks padding)
# --------------------------------------------------------------------------
def remap_pad_to_zero(S, pads=(128, 12, 13)):
    """Per channel: real values -> value+1, pad -> 0. Keeps indices in
    [0, vocab-1] because each channel's max real index is vocab-2."""
    out = np.empty_like(S)
    for ch, pad in enumerate(pads):
        col = S[..., ch]
        out[..., ch] = np.where(col == pad, 0, col + 1)
    return out


# --------------------------------------------------------------------------
# Standard callbacks (report Section 6.4)
# --------------------------------------------------------------------------
def default_callbacks(patience=5, lr_patience=3, min_lr=1e-6):
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=lr_patience, min_lr=min_lr),
    ]
