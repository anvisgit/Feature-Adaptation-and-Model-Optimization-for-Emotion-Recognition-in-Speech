import os

DATASET_PATH = r"c:\Users\sunil\OneDrive\Desktop\SER_RAVDESS_Project"
RESULTS_DIR = "results"
LOGS_DIR = "logs"
MODEL_DIR = "models"
VIZ_DIR = "visualizations"
PROCESSED_DATA_DIR = "processed_data"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

EMOTION_TO_ID = {'01': 0, '02': 1, '03': 2, '04': 3, '05': 4, '06': 5, '07': 6, '08': 7}
ID_TO_EMOTION = {v: k for k, v in EMOTION_TO_ID.items()}
EMOTION_NAMES = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
AUDIO_DURATION = 3.0

N_FOLDS = 5
N_EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.001
RANDOM_SEED = 42
PATIENCE = 30

AUGMENTATION = {
    'noise_factor': 0.005,
    'pitch_steps': 2,
    'speed_range': (0.85, 1.15),
    'num_audio_augments': 4
}

FEATURE_PARAMS = {
    'mfcc': {'nmfcc': 40, 'nfft': 2048, 'hop': 512},
    'mel': {'nmels': 128, 'nfft': 2048, 'hop': 512},
    'gauss': {'window': 25, 'nfft': 2048, 'filters': 64}
}

MIXUP_ALPHA = 0.3
LABEL_SMOOTHING = 0.1
