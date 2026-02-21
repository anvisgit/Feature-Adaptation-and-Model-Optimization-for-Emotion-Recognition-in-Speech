import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization,
    Activation, GlobalAveragePooling1D, Input, Add, GlobalMaxPooling1D, Concatenate,
    SpatialDropout1D, Multiply, Permute, Flatten, RepeatVector, Lambda)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import logging
import io
import numpy as np

logger = logging.getLogger(__name__)

def focalLoss(gamma=1.8, alpha=0.3):
    def focalLossFixed(yTrue, yPred):
        yPred = tf.clip_by_value(yPred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        crossEntropy = -yTrue * tf.math.log(yPred)
        weight = alpha * yTrue * tf.pow(1.0 - yPred, gamma)
        loss = weight * crossEntropy
        return tf.reduce_sum(loss, axis=-1)
    return focalLossFixed

def logModelSummary(model, modelName):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    logger.info(f"{modelName}\n{stream.getvalue()}")

def createBranch(inputs, filters, dropoutRate, name):
    conv3 = Conv1D(filters // 2, 3, padding='same', kernel_initializer='he_normal', name=f'{name}_conv3')(inputs)
    conv5 = Conv1D(filters // 2, 5, padding='same', kernel_initializer='he_normal', name=f'{name}_conv5')(inputs)
    x = Concatenate(name=f'{name}_msConcat')([conv3, conv5])
    x = BatchNormalization(name=f'{name}_bn1')(x)
    x = Activation('relu')(x)
    x = SpatialDropout1D(0.1)(x)

    shortcut = Conv1D(filters * 2, 1, padding='same', name=f'{name}_shortcut')(x)
    x = Conv1D(filters * 2, 3, padding='same', kernel_initializer='he_normal', name=f'{name}_resConv1')(x)
    x = BatchNormalization(name=f'{name}_bn2')(x)
    x = Activation('relu')(x)
    x = SpatialDropout1D(0.15)(x)
    x = Conv1D(filters * 2, 3, padding='same', kernel_initializer='he_normal', name=f'{name}_resConv2')(x)
    x = BatchNormalization(name=f'{name}_bn3')(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(dropoutRate * 0.5)(x)

    attnScore = Dense(1, activation='tanh', name=f'{name}_attnDense')(x)
    attnScore = Flatten(name=f'{name}_attnFlat')(attnScore)
    attnWeight = Activation('softmax', name=f'{name}_attnSoftmax')(attnScore)
    attnWeight = RepeatVector(x.shape[-1])(attnWeight)
    attnWeight = Permute((2, 1))(attnWeight)
    attnOut = Multiply()([x, attnWeight])

    avgPool = GlobalAveragePooling1D(name=f'{name}_gavg')(attnOut)
    maxPool = GlobalMaxPooling1D(name=f'{name}_gmax')(attnOut)
    out = Concatenate(name=f'{name}_poolConcat')([avgPool, maxPool])
    return out

def createMultiBranchModel(mfccShape, melShape, chromaShape, teccShape, numClasses,
                           learningRate=0.0005, filters=96, dropoutRate=0.3,
                           focalGamma=1.8, focalAlpha=0.3):
    logger.info("Building Advanced MultiBranch CNN1D (4-Branch Fusion-Attention)")
    mfccInput = Input(shape=mfccShape, name='mfcc_input')
    melInput = Input(shape=melShape, name='mel_input')
    chromaInput = Input(shape=chromaShape, name='chroma_input')
    teccInput = Input(shape=teccShape, name='tecc_input')

    mfccBranch = createBranch(mfccInput, filters, dropoutRate, 'mfcc')
    melBranch = createBranch(melInput, filters, dropoutRate, 'mel')
    chromaBranch = createBranch(chromaInput, filters // 2, dropoutRate, 'chroma')
    teccBranch = createBranch(teccInput, filters, dropoutRate, 'tecc')

    concat = Concatenate(name='fusion')([mfccBranch, melBranch, chromaBranch, teccBranch])

    dense = Dense(256, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(concat)
    dense = BatchNormalization()(dense)
    dense = Activation('relu')(dense)
    dense = Dropout(dropoutRate)(dense)

    dense = Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(dense)
    dense = BatchNormalization()(dense)
    dense = Activation('relu')(dense)
    dense = Dropout(dropoutRate * 0.7)(dense)

    outputs = Dense(numClasses, activation='softmax', name='output')(dense)

    model = Model([mfccInput, melInput, chromaInput, teccInput], outputs)
    model.compile(optimizer=Adam(learning_rate=learningRate),
                  loss=focalLoss(gamma=focalGamma, alpha=focalAlpha),
                  metrics=['accuracy'])
    logModelSummary(model, "4-Branch Fusion CNN1D with Focal Loss")
    return model

def createCnn1dModel(inputShape, numClasses, learningRate=0.001, filters=128, dropoutRate=0.3):
    inputs = Input(shape=inputShape)
    x = Conv1D(filters, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(numClasses, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learningRate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
