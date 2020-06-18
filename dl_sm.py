import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf

df = pd.read_csv("DAT_MS_EURUSD_M1_202006.csv")


df.columns = ['date', 'heure', 'open', 'high', 'low', 'close', 'nul']


del df['nul']
del df['heure']


del df['date']

X = df[['open','high', 'low']].values
Y = df['close'].values
c = list(zip(X,Y))
random.shuffle(c)
X, Y = zip(*c)
X, Y = np.array(X), np.array(Y)

maximum = max(Y)
num = len(X)//10
Y = Y/maximum
Y1 = Y[:-num]

X1 = X[:-num]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.2)


model = tf.keras.Sequential([
     tf.keras.layers.Dense(50, input_shape = (3,), activation='relu'),
     tf.keras.layers.Dense(50, activation='relu'),
     tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
  ])

model.compile(optimizer='adam',
            loss='mean_squared_error',)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

X_predict = X[-num:]
forecast = model.predict(X_predict)*maximum
forecast = forecast.reshape(num)
Y2 = Y[-num:]*maximum

diff = (forecast - Y2)/forecast
diff*=100
diff = np.abs(diff)
plt.scatter(X_predict[:,0], forecast)
plt.scatter(X_predict[:,0], Y2, c='r', alpha=0.5, linewidths=0.2)
plt.title("Réseau de neurones avec valeur d'ouverture")
plt.legend(["predictions", "Reality"])
plt.xlabel("valeur d'ouverture")
plt.ylabel("valeur de fermeture")
plt.show()

print("Ecart moyen entre la prediction et la réalité en pourcentage :", diff.mean())

#Le réseau de neurone est correct ici car on a en entrée les valeurs d'ouverture, de max et de min qui sont toutes très proches