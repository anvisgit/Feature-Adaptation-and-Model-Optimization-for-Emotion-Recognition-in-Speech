import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Activation
import logging 
from tensorflow.keras.optimizers import Adam
import io

logger=logging.getLogger(__name__)
def log_model_summary(model, model_name):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    logger.info(f"Model Summary for {model_name}:\n{summary_string}")
def create_mlp_model(input_shape, num_classes, learning_rate=0.001, num_units=256, dropout_rate=0.3, num_layers=2):
    model = Sequential()
    model.add(Dense(num_units, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    for _ in range(num_layers - 1):
        model.add(Dense(num_units // 2))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    log_model_summary(model, "MLP")
    return model

