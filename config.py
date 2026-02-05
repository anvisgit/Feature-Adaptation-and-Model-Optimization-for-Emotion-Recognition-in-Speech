import os
import logging

DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"
MODELS_PATH = "models/"
RESULTS_PATH = "results/"
LOGS_PATH = "logs/"

for path in [PROCESSED_DATA_PATH, MODELS_PATH, RESULTS_PATH, LOGS_PATH]:
    os.makedirs(path, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_PATH, "project.log")),
        logging.StreamHandler()
    ]
)

SAMPLE_RATE = 22050
DURATION = 2.5
AUDIO_OFFSET = 0.6
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512

OBSERVED_EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

AUGMENT_NOISE = True
AUGMENT_TIME_STRETCH = True
AUGMENT_PITCH = True

NOISE_AMPLITUDE_FACTOR = 0.035
TIME_STRETCH_RATE = 0.8
PITCH_SHIFT_STEPS = 0.7

EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.15
TEST_SIZE = 0.2
RANDOM_SEED = 42
