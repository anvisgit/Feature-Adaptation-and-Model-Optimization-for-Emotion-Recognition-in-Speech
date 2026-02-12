Speech Emotion Recognition - RAVDESS Dataset
Minimal Research Implementation

Project Structure:
- config.py: Configuration parameters
- features.py: Feature extraction (FBCC, MFCC, TECC)
- dataprep.py: Data loading and preprocessing
- model.py: Neural network architecture
- viz.py: Research visualizations
- train.py: Main training script

Usage:
python train.py

Configuration:
Edit config.py to adjust:
- N_FOLDS: Number of folds (default 5)
- N_EPOCHS: Training epochs (default 30)
- BATCH_SIZE: Batch size (default 32)
- PCA_COMPONENTS: PCA components (default 100)
- MODEL_ARCH: Layer sizes and dropout

Features:
- FBCC (Gaussian): Filter-Based Cepstral Coefficients
- MFCC: Mel-Frequency Cepstral Coefficients
- TECC: Teager Energy Cepstral Coefficients

Output:
- results/: Confusion matrices, accuracy plots, reports
- logs/: Training logs

Model Architecture:
Dense(256) -> Dropout(0.3) -> Dense(128) -> Dropout(0.3) -> Dense(64) -> Dense(8)
Loss: Categorical Cross-Entropy
Optimizer: Adam (lr=0.001)

Stop-Process -Name "python" -Force -ErrorAction SilentlyContinue; Remove-Item -Recurse -Force "c:\Users\sunil\OneDrive\Desktop\SER_RAVDESS_Project\models\*" -ErrorAction SilentlyContinue; Remove-Item -Recurse -Force "c:\Users\sunil\OneDrive\Desktop\SER_RAVDESS_Project\logs\*" -ErrorAction SilentlyContinue; Remove-Item -Recurse -Force "c:\Users\sunil\OneDrive\Desktop\SER_RAVDESS_Project\results\*" -ErrorAction SilentlyContinue; Remove-Item -Recurse -Force "c:\Users\sunil\OneDrive\Desktop\SER_RAVDESS_Project\__pycache__" -ErrorAction SilentlyContinue; Write-Output "Cleaned project directory."

