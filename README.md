# Speech Emotion Recognition (SER) on RAVDESS Dataset

## 📌 Project Overview
This project implements a robust deep learning pipeline for **Speech Emotion Recognition (SER)** using the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset. The system classifies audio recordings into 8 distinct emotional categories: *Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, and Surprised*.

The core objective is to achieve high accuracy (>84%) by leveraging advanced feature extraction techniques, data augmentation, and state-of-the-art neural network architectures.

---

## 🛠️ Technical Architecture

### 1. Feature Extraction Pipeline
Raw audio is not directly fed into models. We extract three types of acoustic features to capture different aspects of speech:

*   **MFCCs (Mel-Frequency Cepstral Coefficients):** 
    *   *What it is:* A representation of the short-term power spectrum of sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency.
    *   *Why use it:* It effectively mimicks the human ear's perception of sound, focusing on frequencies (like speech formants) that carry linguistic and emotional content while filtering out noise.
    *   *Configuration:* 40 coefficients, FFT window size 2048, Hop length 512.

*   **Mel-Spectrograms:**
    *   *What it is:* A spectrogram where the frequencies are converted to the Mel scale. It essentially treats audio as an image where the X-axis is time, Y-axis is frequency (mel-scaled), and color is amplitude (decibels).
    *   *Why use it:* Provides a rich, time-frequency representation that allows Convolutional Neural Networks (CNNs) to learn patterns (like rising pitch in anger vs. falling pitch in sadness) just like they process images.
    *   *Configuration:* 128 Mel bands.

*   **Gaussian Filterbank Features:**
    *   *What it is:* Features derived by applying Gaussian filters to the power spectrum, followed by a Discrete Cosine Transform (DCT).
    *   *Why use it:* A specialized feature set that can capture spectral envelope characteristics often missed by standard MFCCs.

*   **Statistical Functionals (For MLP):**
    *   Since Multi-Layer Perceptrons (MLPs) cannot handle variable-length sequences well, we condense the time-series data into a fixed-size vector by computing 13 statistical moments per feature (e.g., Mean, Std Dev, Max, Min, Skewness, Kurtosis, Percentiles). This results in a comprehensive "global" summary of the emotion in the clip.

### 2. Data Augmentation (Addressing Small Data)
RAVDESS is a small dataset (~1440 samples). Deep learning models are prone to overfitting on small data. We solve this using **4x Audio-Domain Augmentation**:
For every training fold, we create 4 altered copies of each sample on the fly:
*   **Time Stretching:** Speeding up or slowing down speech (0.8x - 1.2x) without changing pitch.
*   **Pitch Shifting:** Raising or lowering the pitch (±2 semitones) without changing speed.
*   **Additive Noise:** Injecting low-level Gaussian noise to make the model robust to imperfect recording conditions.

> **Why this matters:** This artificially expands our training set from ~1100 to ~5500 samples per fold, forcing the model to learn the *concept* of an emotion rather than memorizing the specific raw audio files.

---

## 🧠 Model Architectures

We implemented and compared four distinct deep learning architectures:

### 1. CNN-1D (Convolutional Neural Network)
*   **Concept:** Adapts image recognition principles to audio. It slides 1-dimensional filters over the time axis of the MFCC/Mel-spectrogram sequence.
*   **Key Innovation:** Uses **Residual Blocks** (ResNet style). Each block has a "shortcut connection" that allows gradients to flow through the network easily, preventing the "vanishing gradient" problem and allowing us to train deeper networks effectively.
*   **Structure:** 
    *   Input -> Conv1D -> Residual Blocks -> Global Average & Max Pooling -> Dense -> Softmax.
    *   **Pooling:** We use both *Global Average* and *Global Max* pooling concatenated together to capture both the *average* emotional tone and the *peak* emotional intensity.

### 2. LSTM (Long Short-Term Memory)
*   **Concept:** A type of Recurrent Neural Network (RNN) designed for sequential data. It has "memory cells" that can maintain context over time.
*   **Key Innovation:** **Bidirectional Processing**. The model processes the audio from start-to-end AND end-to-start simultaneously. This means at any point in time, the network knows what was said before *and* what is coming next, providing full context.
*   **Structure:** 2 Bidirectional LSTM layers followed by Dense layers.

### 3. Hybrid Model (CNN + LSTM)
*   **Concept:** Best of both worlds. 
*   **How it works:** The CNN layers act as a feature extractor, identifying local patterns (like a sharp intake of breath) in the spectrogram. The output of the CNN is then fed into an LSTM to understand how these patterns evolve over time.
*   **Why:** CNNs are fast and good at local features; LSTMs are good at long-term temporal dependencies.

### 4. MLP (Multi-Layer Perceptron)
*   **Concept:** A classic feed-forward neural network.
*   **Input:** The "Statistical Functionals" vector described above (global statistics).
*   **Use Case:** Serves as a strong baseline to compare against the sequence models (CNN/LSTM). If the temporal models don't outperform the MLP, it means the sequence information isn't being used effectively.

---

## 🔬 Training Strategy

### Cross-Validation
*   **Method:** **5-Fold Stratified Cross-Validation**.
*   **Why:** Instead of a single train/test split, we rotate the data so every sample is used for both training and testing. "Stratified" ensures each fold has the same percentage of each emotion (e.g., 15% Happy, 15% Sad) to preventing bias.

### Regularization Techniques (Preventing Overfitting)
1.  **Mixup:** A state-of-the-art technique where we train on linear combinations of pairs of examples and their labels. Instead of telling the model "This is Happy", we generate a sample that is "70% Happy + 30% Sad" and force the model to predict that ratio. This leads to smoother decision boundaries.
2.  **Label Smoothing:** Instead of targeting strict 0 or 1 labels (which causes overconfidence), we target 0.1 and 0.9. This prevents the model from becoming too arrogant about its predictions.
3.  **Early Stopping:** Monitoring validation accuracy. If the model stops improving for 30 epochs, we stop training and revert to the best weights.
4.  **ReduceLROnPlateau:** If learning stalls, we automatically reduce the learning rate by half to fine-tune the weights.

---

## 📊 Performance & Evaluation
*   **Primary Metric:** **Accuracy** (Percentage of correct predictions).
*   **Secondary Metric:** **F1-Score** (Weighted average of Precision and Recall, crucial if classes are imbalanced).
*   **Confusion Matrix:** Shows exactly which emotions are being confused (e.g., distinguishing "Anger" vs "Disgust" is often difficult).

---

## 🚀 How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Dataset:** Ensure RAVDESS `Actor_*` folders are in the project root.
3.  **Train Models:**
    ```bash
    python train.py
    ```
    *This runs the full 5-fold CV for all models and features. It may take several hours.*
4.  **Visualize Results:**
    ```bash
    python viz.py
    ```
    *Generates plots in the `visualizations/` directory.*

## 📁 Project Structure
*   `config.py`: Global settings (sample rate, FFT size, learning rate).
*   `data_preprocessing.py`: Handles loading, augmentation, and feature normalization.
*   `features.py`: Implementations of MFCC, Mel-Spec, and Gauss extraction.
*   `model.py`: TensorFlow/Keras definitions of CNN, LSTM, Hybrid, MLP.
*   `train.py`: Main training loop with Cross-Validation and logging.
*   `viz.py`: Generates confusion matrices, loss curves, and dataset stats.
