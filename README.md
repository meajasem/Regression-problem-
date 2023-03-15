# Regression-problem-
I am learning regression problems and im facing an issue that i can figure out. my model predicts the same value every time, no mater the input. 
this is the code:- 

!pip install -q seaborn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

dataset = pd.read_csv('VideoGamesSales.csv') 
dataset

set2 = dataset.drop(["NA_Sales","EU_Sales","JP_Sales","Other_Sales","Critic_Score","Critic_Count","User_Score","User_Count","Developer","Rating"],axis=1) 
set2

from sklearn.preprocessing import LabelEncoder 
# integer encode
label_encoder = LabelEncoder() 

set2.Name = label_encoder.fit_transform(set2.Name)
set2.Platform = label_encoder.fit_transform(set2.Platform)
set2.Genre = label_encoder.fit_transform(set2.Genre)
set2.Publisher = label_encoder.fit_transform(set2.Publisher)

set2.isna().sum()
set22 = set2.dropna()

from sklearn.model_selection import train_test_split
x= set22[['Name','Platform','Year_of_Release','Genre','Publisher']]
y= set22['Global_Sales']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.4, random_state=3)

model = keras.Sequential([
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1, activation='linear')
])

model.compile(loss='mean_absolute_error',
              optimizer=tf.keras.optimizers.Adam(0.001))
              
model.fit(
    x_train,
    y_train,
    epochs=600)
    
model.predict([[5438.0,29.0,2004.0,6.0,491.0]])

