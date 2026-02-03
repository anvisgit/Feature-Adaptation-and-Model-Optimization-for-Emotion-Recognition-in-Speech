import os 
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScalar , LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, classification_report
import tensorflow as tf
import config 
import data_preprocessing as dp 
import models 
import visualizations as viz 
import logging 

logger= logging.getlogger(__name__)
np.random.seed(42)
tf.random.set_seed(42)

def prepare_data():
  logger.info("Preparing data")
  x,y=dp.load_data()
  if x.shape[0]==0:
    raise ValueError("No data found")
  y=LabelEncoder.fit_transform(y)
  xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.3, random_state=42)
  sc=StandardScaler()
  xtrain=sc.fit_transform(xtrain)
  xtest=sc.fit_transform(xtest)
  joblib.dump(sc, os.path.join(config.MODELS_PATH, 'scaler.pkl'))
  joblib.dump(LabelEncoder(), os.path.join(config.MODELS_PATH, 'label_encoder.pkl'))
  logger.info(f"Data split. Train: {trainx.shape}, Test: {testx.shape}")
  logger.info(f"Classes encoded: {list(zip(range(len(le.classes_)), le.classes_))}")

  return xtrain,xtest,ytrain,ytest, LabelEncoder

  def reshape1(x): # reshape for cnn lstm ;2D->3D
    return x.reshape(x.shape[0], x.shape[1], 1)
  def evaluate():

  if __name__=="__main__":
    evaluate()
  
    

    
