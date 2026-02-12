import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization, Activation, GlobalAveragePooling1D, LSTM, Bidirectional, Input, Add, GlobalMaxPooling1D, Concatenate, SpatialDropout1D
import logging
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import io
import numpy as np

logger = logging.getLogger(__name__)

def logModelSummary(model, modelName):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    logger.info(f"Model Summary for {modelName}:\n{stream.getvalue()}")

def createMlpModel(inputShape, numClasses, learningRate=0.001, dropoutRate=0.5):
    logger.info(f"Creating MLP Model. Input: {inputShape}")
    model = Sequential([
        Input(shape=inputShape),
        Dense(512, kernel_initializer='he_normal', kernel_regularizer=l2(1e-3)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(dropoutRate),
        Dense(256, kernel_initializer='he_normal', kernel_regularizer=l2(1e-3)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(dropoutRate),
        Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(1e-3)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(dropoutRate * 0.6),
        Dense(numClasses, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learningRate), loss='categorical_crossentropy', metrics=['accuracy'])
    logModelSummary(model, "MLP")
    return model

def createCnn1dModel(inputShape, numClasses, learningRate=0.001, filters=64, dropoutRate=0.3):
    logger.info(f"Creating CNN-1D Model. Input: {inputShape}")
    inputs = Input(shape=inputShape)
    x = Conv1D(filters, 5, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout1D(0.1)(x)

    for f in [filters * 2, filters * 4]:
        shortcut = Conv1D(f, 1, padding='same')(x)
        x = Conv1D(f, 3, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv1D(f, 3, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(dropoutRate)(x)

    avgPool = GlobalAveragePooling1D()(x)
    maxPool = GlobalMaxPooling1D()(x)
    concat = Concatenate()([avgPool, maxPool])
    dense = Dense(256, kernel_initializer='he_normal', kernel_regularizer=l2(1e-3))(concat)
    dense = BatchNormalization()(dense)
    dense = Activation('relu')(dense)
    dense = Dropout(dropoutRate + 0.1)(dense)
    outputs = Dense(numClasses, activation='softmax')(dense)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learningRate), loss='categorical_crossentropy', metrics=['accuracy'])
    logModelSummary(model, "CNN-1D")
    return model

def createLstmModel(inputShape, numClasses, learningRate=0.001, units=64, dropoutRate=0.3):
    logger.info(f"Creating LSTM Model. Input: {inputShape}")
    model = Sequential([
        Input(shape=inputShape),
        Bidirectional(LSTM(units, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)),
        BatchNormalization(),
        Bidirectional(LSTM(units, dropout=0.2, recurrent_dropout=0.1)),
        BatchNormalization(),
        Dense(128, activation='relu', kernel_regularizer=l2(1e-3)),
        Dropout(dropoutRate),
        Dense(numClasses, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learningRate), loss='categorical_crossentropy', metrics=['accuracy'])
    logModelSummary(model, "LSTM")
    return model

def createHybridModel(inputShape, numClasses, learningRate=0.001, filters=64, lstmUnits=64, dropoutRate=0.3):
    logger.info(f"Creating Hybrid Model. Input: {inputShape}")
    inputs = Input(shape=inputShape)
    x = Conv1D(filters, 5, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(filters * 2, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)

    x = Bidirectional(LSTM(lstmUnits, return_sequences=False, dropout=0.2, recurrent_dropout=0.1))(x)
    x = BatchNormalization()(x)
    dense = Dense(128, activation='relu', kernel_regularizer=l2(1e-3))(x)
    dense = Dropout(dropoutRate)(dense)
    outputs = Dense(numClasses, activation='softmax')(dense)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learningRate), loss='categorical_crossentropy', metrics=['accuracy'])
    logModelSummary(model, "Hybrid")
    return model
