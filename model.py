import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization,
    Activation, GlobalAveragePooling1D, LSTM, Bidirectional, Input, Add,
    GlobalMaxPooling1D, Concatenate, SpatialDropout1D, Multiply, Reshape,
    Permute, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Rescaling)
from tensorflow.keras.applications import EfficientNetB0
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

def buildBranch(inputTensor, name, filters=[64, 64, 128, 128, 256]):
    logger.info(f"Building 1D-CNN branch '{name}' with filters {filters}")
    x = inputTensor
    for i, f in enumerate(filters):
        x = Conv1D(f, 3, padding='same', kernel_initializer='he_normal', name=f'{name}Conv{i+1}')(x)
        x = BatchNormalization(name=f'{name}Bn{i+1}')(x)
        x = Activation('relu', name=f'{name}Relu{i+1}')(x)
    x = GlobalAveragePooling1D(name=f'{name}Gap')(x)
    logger.info(f"Branch '{name}' output shape after GAP: {x.shape}")
    return x

def createMultiBranchModel(mfccShape, melShape, chromaShape, numClasses, learningRate=0.001, dropoutRate=0.4):
    logger.info(f"Creating Multi-Branch 1D-CNN Fusion Model")
    logger.info(f"Input shapes: MFCC={mfccShape} Mel={melShape} Chroma={chromaShape}")

    mfccInput = Input(shape=mfccShape, name='mfccInput')
    melInput = Input(shape=melShape, name='melInput')
    chromaInput = Input(shape=chromaShape, name='chromaInput')

    logger.info("Building MFCC branch with 5 Conv1D layers")
    mfccBranch = buildBranch(mfccInput, 'mfcc', filters=[64, 64, 128, 128, 256])

    logger.info("Building Mel branch with 5 Conv1D layers")
    melBranch = buildBranch(melInput, 'mel', filters=[64, 64, 128, 128, 256])

    logger.info("Building Chroma branch with 5 Conv1D layers")
    chromaBranch = buildBranch(chromaInput, 'chroma', filters=[32, 32, 64, 64, 128])

    logger.info("Concatenating all three branch outputs for learnable fusion")
    merged = Concatenate(name='fusion')([mfccBranch, melBranch, chromaBranch])

    x = Dense(512, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), name='fusionDense1')(merged)
    x = BatchNormalization(name='fusionBn1')(x)
    x = Activation('relu', name='fusionRelu1')(x)
    x = Dropout(dropoutRate, name='fusionDrop1')(x)

    x = Dense(256, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), name='fusionDense2')(x)
    x = BatchNormalization(name='fusionBn2')(x)
    x = Activation('relu', name='fusionRelu2')(x)
    x = Dropout(dropoutRate, name='fusionDrop2')(x)

    x = Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), name='fusionDense3')(x)
    x = BatchNormalization(name='fusionBn3')(x)
    x = Activation('relu', name='fusionRelu3')(x)
    x = Dropout(dropoutRate * 0.5, name='fusionDrop3')(x)

    outputs = Dense(numClasses, activation='softmax', name='output')(x)

    model = Model([mfccInput, melInput, chromaInput], outputs)
    model.compile(
        optimizer=Adam(learning_rate=learningRate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    logModelSummary(model, "MultiBranch-1D-CNN-Fusion")
    return model

def createMlpModel(inputShape, numClasses, learningRate=0.001, dropoutRate=0.3):
    logger.info(f"Creating MLP Model with input shape {inputShape}")
    model = Sequential([
        Input(shape=inputShape),
        Dense(512, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)),
        BatchNormalization(), Activation('relu'), Dropout(dropoutRate),
        Dense(256, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)),
        BatchNormalization(), Activation('relu'), Dropout(dropoutRate),
        Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)),
        BatchNormalization(), Activation('relu'), Dropout(dropoutRate * 0.6),
        Dense(numClasses, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learningRate), loss='categorical_crossentropy', metrics=['accuracy'])
    logModelSummary(model, "MLP")
    return model

def createCnn1dModel(inputShape, numClasses, learningRate=0.001, filters=128, dropoutRate=0.3):
    logger.info(f"Creating CNN-1D Model with input shape {inputShape}")
    inputs = Input(shape=inputShape)
    conv3 = Conv1D(filters // 2, 3, padding='same', kernel_initializer='he_normal')(inputs)
    conv5 = Conv1D(filters // 2, 5, padding='same', kernel_initializer='he_normal')(inputs)
    x = Concatenate()([conv3, conv5])
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
    attnScore = Dense(1, activation='tanh')(x)
    attnScore = tf.keras.layers.Flatten()(attnScore)
    attnWeight = Activation('softmax')(attnScore)
    attnWeight = tf.keras.layers.RepeatVector(x.shape[-1])(attnWeight)
    attnWeight = Permute((2, 1))(attnWeight)
    attnOut = Multiply()([x, attnWeight])
    avgPool = GlobalAveragePooling1D()(attnOut)
    maxPool = GlobalMaxPooling1D()(attnOut)
    concat = Concatenate()([avgPool, maxPool])
    dense = Dense(256, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(concat)
    dense = BatchNormalization()(dense)
    dense = Activation('relu')(dense)
    dense = Dropout(dropoutRate + 0.1)(dense)
    dense = Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(dense)
    dense = BatchNormalization()(dense)
    dense = Activation('relu')(dense)
    dense = Dropout(dropoutRate)(dense)
    outputs = Dense(numClasses, activation='softmax')(dense)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learningRate), loss='categorical_crossentropy', metrics=['accuracy'])
    logModelSummary(model, "CNN-1D")
    return model

def createLstmModel(inputShape, numClasses, learningRate=0.001, units=128, dropoutRate=0.3):
    logger.info(f"Creating LSTM Model with input shape {inputShape}")
    inputs = Input(shape=inputShape)
    x = Bidirectional(LSTM(units, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(inputs)
    x = BatchNormalization()(x)
    x = Bidirectional(LSTM(units, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(x)
    x = BatchNormalization()(x)
    attnScore = Dense(1, activation='tanh')(x)
    attnScore = tf.keras.layers.Flatten()(attnScore)
    attnWeight = Activation('softmax')(attnScore)
    attnWeight = tf.keras.layers.RepeatVector(x.shape[-1])(attnWeight)
    attnWeight = Permute((2, 1))(attnWeight)
    attnOut = Multiply()([x, attnWeight])
    x = GlobalAveragePooling1D()(attnOut)
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(dropoutRate)(x)
    outputs = Dense(numClasses, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learningRate), loss='categorical_crossentropy', metrics=['accuracy'])
    logModelSummary(model, "LSTM")
    return model

def createHybridModel(inputShape, numClasses, learningRate=0.001, filters=64, lstmUnits=128, dropoutRate=0.3):
    logger.info(f"Creating Hybrid Model with input shape {inputShape}")
    inputs = Input(shape=inputShape)
    x = Conv1D(filters, 5, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(filters * 2, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(filters * 4, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    sePool = GlobalAveragePooling1D()(x)
    se = Dense(filters * 4 // 4, activation='relu')(sePool)
    se = Dense(filters * 4, activation='sigmoid')(se)
    se = Reshape((1, filters * 4))(se)
    x = Multiply()([x, se])
    x = MaxPooling1D(2)(x)
    x = Bidirectional(LSTM(lstmUnits, return_sequences=False, dropout=0.2, recurrent_dropout=0.1))(x)
    x = BatchNormalization()(x)
    dense = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    dense = Dropout(dropoutRate)(dense)
    outputs = Dense(numClasses, activation='softmax')(dense)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learningRate), loss='categorical_crossentropy', metrics=['accuracy'])
    logModelSummary(model, "Hybrid")
    return model

def createEfficientNetModel(inputShape, numClasses, learningRate=0.001, dropoutRate=0.5):
    logger.info(f"Creating EfficientNetB0 2D Model with input shape {inputShape}")
    inputs = Input(shape=inputShape)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    baseModel = EfficientNetB0(weights='imagenet', include_top=False, input_shape=inputShape, pooling='avg')
    baseModel.trainable = False
    x = baseModel(x, training=False)
    x = Dense(256, kernel_initializer='he_normal', kernel_regularizer=l2(1e-3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(1e-3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropoutRate)(x)
    outputs = Dense(numClasses, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learningRate), loss='categorical_crossentropy', metrics=['accuracy'])
    logModelSummary(model, "EfficientNetB0-2D")
    return model

def createCustomCnn2dModel(inputShape, numClasses, learningRate=0.001, dropoutRate=0.3):
    logger.info(f"Creating Custom CNN-2D Model with input shape {inputShape}")
    inputs = Input(shape=inputShape)
    x = Rescaling(1.0 / 255.0)(inputs)
    for f, d in [(32, 0.1), (64, 0.15), (128, 0.2), (256, 0.25)]:
        x = Conv2D(f, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(f, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(d)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropoutRate * 0.6)(x)
    outputs = Dense(numClasses, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learningRate), loss='categorical_crossentropy', metrics=['accuracy'])
    logModelSummary(model, "Custom-CNN-2D")
    return model
