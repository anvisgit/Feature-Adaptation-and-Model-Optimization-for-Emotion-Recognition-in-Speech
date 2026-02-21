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
N_MFCC = 40
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
AUDIO_DURATION = 4.0

MEL_IMG_SIZE = 224

N_FOLDS = 5
N_EPOCHS = 150
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
RANDOM_SEED = 42
PATIENCE = 15

FOCAL_GAMMA = 1.8
FOCAL_ALPHA = 0.3

AUGMENTATION = {
    'noiseFactor': 0.005,
    'pitchSteps': 2,
    'speedRange': (0.85, 1.15),
    'numAudioAugments': 4
}

FEATURE_PARAMS = {
    'mfcc': {'nmfcc': 40, 'nfft': 2048, 'hop': 512},
    'mel': {'nmels': 128, 'nfft': 2048, 'hop': 512},
    'gauss': {'window': 25, 'nfft': 2048, 'filters': 64},
    'mel2d': {'nmels': 128, 'nfft': 2048, 'hop': 512, 'imgSize': MEL_IMG_SIZE},
    'multi': {'nmfcc': 40, 'nmels': 128, 'nfft': 2048, 'hop': 512, 'ntecc': 40}
}

MIXUP_ALPHA = 0.2
