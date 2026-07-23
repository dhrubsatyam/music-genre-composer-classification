# Music Genre & Composer Classification Using Deep Learning

## 📌 Project Overview
Music is a ubiquitous art form with a rich historical legacy. Classical composers developed distinct stylistic signatures that define their works. However, attributing a specific musical piece to its correct composer can be challenging for novice listeners and musicians.

This project uses deep learning architectures—specifically **Long Short-Term Memory (LSTM)** networks and **Convolutional Neural Networks (CNN)**—to automatically analyze musical scores in MIDI format and accurately classify the composer among four major classical icons:
1. **Johann Sebastian Bach**
2. **Ludwig van Beethoven**
3. **Frédéric Chopin**
4. **Wolfgang Amadeus Mozart**

---

## 👨‍🏫 Course & Group Information
* **Course:** AAI 511 – Applied AI / Deep Learning
* **Group:** Group 11
* **Under the Guidance of: Prof. Premkumar Chithaluru, Ph.D.**

### 👥 Team Members
* **Harshil Shah**
* **Dhrub Satyam**
* **Yatharth Vardan**

---

## 🎯 Objectives
* Convert raw MIDI musical scores into structured temporal and spectral representations suitable for neural networks.
* Extract key musical parameters including **notes, pitch, duration, chords, and tempo**.
* Design, train, and evaluate **CNN** and **LSTM** models for multi-class composer classification.
* Fine-tune hyperparameters to maximize model accuracy, precision, and recall.

---

## 📊 Dataset
The dataset consists of MIDI files collected from classical musical scores.
* **Target Composers:** Bach, Beethoven, Chopin, and Mozart.
* **Source:** [Kaggle Classical Music MIDI Dataset](https://www.kaggle.com/)

> **Note:** Raw dataset files are excluded from git tracking via `.gitignore` due to repository size constraints.

---

## ⚙️ Methodology & Pipeline

The project execution is organized sequentially across Jupyter Notebooks in the `notebooks/` directory:

1. **`01_data_preprocessing.ipynb`**
   * Filter dataset to restrict entries strictly to the four target composers.
   * Parse MIDI files and perform necessary data cleaning and augmentation.
2. **`02_feature_extraction.ipynb`**
   * Extract features such as pitch sequences, velocity, durations, chord structures, and tempo changes using MIDI processing libraries (`pretty_midi`, `mido`, or `music21`).
3. **`03_model_building_and_evaluation.ipynb`**
   * **CNN Model:** Captures spatial/spectral features from piano-roll representations or pitch spectrograms.
   * **LSTM Model:** Captures sequential, time-series relationships across musical note progressions.
   * Evaluate models using **Accuracy**, **Precision**, **Recall**, and **F1-Score**.

---

## 📁 Repository Structure
```text
.
├── data/
│   ├── raw/                                 # Raw MIDI files (git-ignored)
│   └── processed/                           # Extracted features & datasets (git-ignored)
├── docs/                                    # Final project report PDF
├── notebooks/                               # Sequential Jupyter Notebooks
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_feature_extraction.ipynb
│   └── 03_model_building_and_evaluation.ipynb
├── .gitignore
├── README.md
└── requirements.txt                         # Dependencies