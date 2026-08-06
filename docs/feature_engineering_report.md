# Feature Engineering Report — Composer Classification

**Notebook:** `notebooks/02_feature_extraction.ipynb`
**Task:** AAI 511 Final Project · Group 11
**Composers (classes):** Bach · Beethoven · Chopin · Mozart
**Author of this stage:** Yatharth Vardan

---

## 1. Purpose and scope

This stage converts each cleaned MIDI file into the numerical tensor
representations consumed by the three modelling approaches — **CNN**, **LSTM**,
and a **hybrid ensemble**. It sits between `01_data_preprocessing.ipynb`
(corpus download + NLP-style vectorisation) and
`03_model_building_and_evaluation.ipynb` (training + evaluation), and produces
cached NumPy tensors so the modelling notebook never re-parses MIDI.

## 2. Relationship to notebook 01 (important context)

Notebook 01 treated symbolic music as *text*: it tokenised notes/chords and
produced **document-level** vectors via Bag-of-Words, TF-IDF, and **Word2Vec**.
Those representations are a single fixed-length vector *per file* and therefore
**cannot** feed a convolutional or recurrent network, which need spatial (image)
and sequential inputs respectively.

This notebook therefore **re-parses the raw MIDI** to build the spatial and
sequential tensors the deep nets require, while **carrying notebook 01's
Word2Vec (CBOW, 100-d) vectors forward** — aligned by file — as a third,
complementary feature set for a classical-ML branch inside the hybrid ensemble.
It only borrows the *file list and labels* from notebook 01 so that every
representation stays row-aligned with the same samples.

## 3. Libraries

| Library | Role |
|---|---|
| `pretty_midi` | Note-level parsing (pitch, onset, offset, velocity), tempo & time-signature access, piano-roll math |
| `music21` | Higher-level harmonic analysis — key estimation (EDA §10) |
| `mido` | Low-level MIDI fallback (imported for completeness) |
| `numpy` / `pandas` / `matplotlib` | Tensor assembly, tabular EDA, visualisation |

## 4. MIDI parsing (report §5.1)

Each file's instrument tracks are flattened into one ordered list of note
events `(onset, offset, pitch, velocity)`. Design choices:

- **Drum tracks dropped** — unpitched, not informative for composer style.
- **Overlapping notes retained** — preserves simultaneous/chordal structure.
- **Tempo resolved best-effort** — first tempo change → `estimate_tempo()` →
  fallback `120 BPM`. Tempo is needed to place events on a musical grid.

## 5. Extracted musical parameters (report §5.2)

A per-file **musical-parameter summary** is computed (saved to
`data/processed/file_summary.csv`) for transparency and EDA:

| Parameter | How it is measured |
|---|---|
| Pitch sequence / range | ordered MIDI pitch numbers; min/max/range |
| Note duration | offset − onset, later quantised to 16th-note units |
| Velocity / dynamics | MIDI velocity 0–127; per-file mean |
| Chord structure | notes sharing a near-identical onset are grouped; count + ratio |
| Tempo & time signature | first tempo change (BPM) and first time-signature event |

These describe the corpus and motivate the tensor designs; only tempo and time
shape the grid — the rest are descriptive.

## 6. Input representations (report §5.3)

### 6.1 The paired-window design (key decision)

Every file is placed on a **tempo-normalised 16th-note grid** (4 steps per beat)
and sliced into **fixed-length overlapping time windows** ("excerpts"), default
**128 steps ≈ 8 bars in 4/4**, hop 64 (50% overlap), capped at 40 windows/file.

Each window emits **both** representations from the **same slice of music**, so
the CNN and LSTM datasets are **row-aligned**. This is what lets the hybrid
ensemble fuse the two branches per sample. Windowing also serves as light data
augmentation — essential given only ~80 files/composer.

Rationale for the 16th-note grid rather than wall-clock seconds: it makes fast
and slow performances of the same music comparable, so the models learn
composer style rather than performance tempo.

### 6.2 Piano-roll matrix — CNN input

- Shape **`128 × 128`** (pitch × time-steps), `float32`.
- **Velocity-weighted**: `roll[p, t] = velocity/127` when pitch `p` sounds at
  step `t`, else 0 — richer than binary and encodes dynamics.
- Image-like input for 2-D convolutions that detect chord voicings, parallel
  motion, and textural density. Modelling adds a channel dim → `(N,128,128,1)`.

### 6.3 Encoded note sequence — LSTM input

- Shape **`128 × 3`** (`SEQ_LEN` notes × 3 channels), `int64`; padded/truncated.
- Channels per note: **`(pitch, dur_bin, ioi_bin)`**
  - `pitch`: MIDI 0–127, index 128 = pad
  - `dur_bin`: note length in 16th-note units, bucketed by `DUR_BINS`
  - `ioi_bin`: inter-onset interval to next note (16th units), bucketed by
    `IOI_BINS`; an IOI ≈ 0 marks notes struck together (a chord)
- Continuous rhythm is **quantised into a small ordinal vocabulary** so each
  channel can go through an embedding layer and generalise across tempi.

Quantisation edges (16th-note units):

```
DUR_BINS = [0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
IOI_BINS = [0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
```

Embedding vocab sizes (in metadata): `pitch = 129`, `dur = 13`, `ioi = 14`.

### 6.4 Word2Vec document vector — classical / hybrid branch

- Shape **`100`** per file (CBOW, from notebook 01), broadcast to every excerpt
  of that file so it stays row-aligned with the other two arrays.
- If notebook 01's artifacts are absent, this branch is filled with zeros and a
  warning is printed (the deep branches still work).

## 7. Leakage-safe alignment (critical)

Each excerpt records a **`group` id = its source file index**. Excerpts from one
piece are highly correlated, so the train/test split in notebook 03 **must be
grouped by file** (`StratifiedGroupKFold` / `GroupShuffleSplit`). Splitting
excerpts randomly would leak a piece across train and test and inflate accuracy.
The extraction guarantees each group carries a single composer label.

## 8. Outputs (cached to `data/processed/`)

| File | Contents |
|---|---|
| `features.npz` | `X_pianoroll (N,128,128)`, `X_sequence (N,128,3)`, `X_w2v (N,100)`, `y (N,)`, `groups (N,)`, `composer_classes (4,)` |
| `feature_metadata.json` | grid, windowing, bin edges, vocab sizes, shapes, class order, W2V-enabled flag |
| `file_summary.csv` | one row per file with the §5.2 musical parameters |

All arrays share the same first dimension `N` (number of excerpts) and are
row-aligned by construction.

## 9. Robustness & reproducibility

- **Runs with or without notebook 01's artifacts** — primary path loads the
  file list + Word2Vec vectors; fallback rediscovers MIDI under `data/raw/`
  with the same 80-files/composer cap and seed, then disables the W2V branch.
- **Path resolution** relocates files by basename under `data/raw/` so paths
  saved on another machine (e.g. Kaggle) still resolve locally.
- **Fixed seed (42)** and pure quantisation make extraction deterministic.
- Parse failures / empty files are skipped and reported, never silently dropped.

## 10. Validation performed

The notebook's code was executed end-to-end on a synthetic MIDI corpus (both the
nb01-present and nb01-absent paths) and verified:

- Correct tensor shapes and dtypes; all arrays row-aligned.
- Chords preserved (simultaneous notes → IOI ≈ 0 in sequences; stacked pitches
  in the piano roll).
- Word2Vec vectors correctly broadcast to each file's excerpts.
- Every `group` carries exactly one label (no label mixing).
- `music21` key estimation returns sensible keys.
- Cached `.npz` reloads with identical shapes; metadata JSON is valid.

## 11. Handoff to modelling (notebook 03)

```python
d = np.load("data/processed/features.npz", allow_pickle=True)
X_pianoroll, X_sequence, X_w2v = d["X_pianoroll"], d["X_sequence"], d["X_w2v"]
y, groups = d["y"], d["groups"]
```

| Model | Input | Notes |
|---|---|---|
| CNN | `X_pianoroll` → `(N,128,128,1)` | 2-D convs over pitch × time |
| LSTM | `X_sequence (N,128,3)` | embed each channel (vocabs in metadata), concat, mask pad |
| Hybrid | CNN feats ⊕ LSTM feats ⊕ `X_w2v` | late fusion → dense classification head |

Split with `groups`; classes are mildly imbalanced, so use stratification or
`class_weight`.

## 12. Tunable parameters (for later hyperparameter sweeps)

`PR_WINDOW`, `PR_HOP`, `MAX_WINDOWS`, `SEQ_LEN`, `STEPS_PER_BEAT`, and the
`DUR_BINS` / `IOI_BINS` edges are all defined in one config cell (§1.3). Shorter
windows + smaller hop → more, more-augmented excerpts; larger windows → more
long-range context per sample.
