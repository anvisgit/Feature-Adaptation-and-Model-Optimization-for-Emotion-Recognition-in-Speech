import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Activation, GlobalAveragePooling1D
import logging
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import io

logger = logging.getLogger(__name__)

def log_model_summary(model, model_name):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    logger.info(f"Model Summary for {model_name}:\n{summary_string}")

def create_mlp_model(input_shape, num_classes, learning_rate=0.0005, num_units=512, dropout_rate=0.4, num_layers=3):
    logger.info(f"Creating MLP Model. Layers: {num_layers}, Units: {num_units}, Dropout: {dropout_rate}")
    model = Sequential()
    
    model.add(Dense(num_units, input_shape=input_shape, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    
    for i in range(num_layers - 1):
        units = num_units // (2 ** (i + 1))
        model.add(Dense(units, kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    log_model_summary(model, "MLP")
    return model

def create_cnn1d_model(input_shape, num_classes, learning_rate=0.0005, filters=128, kernel_size=5, dropout_rate=0.4):
    logger.info(f"Creating CNN-1D Model. Filters: {filters}, Kernel: {kernel_size}")
    model = Sequential()
    
    model.add(Conv1D(filters, kernel_size, padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))
    
    model.add(Conv1D(filters*2, kernel_size, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))
    
    model.add(Conv1D(filters*3, kernel_size//2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    log_model_summary(model, "CNN-1D")
    return model

def create_lstm_model(input_shape, num_classes, learning_rate=0.0005, units=256, dropout_rate=0.4):
    logger.info(f"Creating LSTM Model. Units: {units}")
    from tensorflow.keras.layers import LSTM, Bidirectional

    model = Sequential()
    
    model.add(Bidirectional(LSTM(units, return_sequences=True, kernel_regularizer=l2(0.001)), input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    
    model.add(Bidirectional(LSTM(units // 2, return_sequences=True, kernel_regularizer=l2(0.001))))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    
    model.add(Bidirectional(LSTM(units // 4, kernel_regularizer=l2(0.001))))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    log_model_summary(model, "LSTM")
    return model

def create_hybrid_model(input_shape, num_classes, learning_rate=0.0003, filters=128, lstm_units=128, dropout_rate=0.5):
    logger.info(f"Creating Hybrid CNN-LSTM Model. Filters: {filters}, LSTM Units: {lstm_units}")
    from tensorflow.keras.layers import LSTM, Bidirectional
    
    model = Sequential()
    
    model.add(Conv1D(filters, 5, padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate * 0.6))
    
    model.add(Conv1D(filters*2, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate * 0.6))
    
    model.add(Bidirectional(LSTM(lstm_units, return_sequences=True)))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    
    model.add(Bidirectional(LSTM(lstm_units // 2)))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(dropout_rate * 0.7))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    log_model_summary(model, "Hybrid-CNN-LSTM")
    return model
