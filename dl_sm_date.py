import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
from datetime import datetime
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf

df = pd.read_csv("DAT_MS_EURUSD_M1_202006.csv")


df.columns = ['date', 'heure', 'open', 'high', 'low', 'close', 'nul']



Y = df['close'].values
x = list(enumerate(Y))
random.shuffle(x)
x, Y = zip(*x)
x, Y = np.array(x), np.array(Y)


num = len(x)//100
Y = Y
Y1 = Y[:-num]
Y2 = Y[-num:]
max_y1 = max(Y1)
max_y2 = max(Y2)
Y1=Y1/max_y1

X1 = x[:-num]
max_x1 = max(X1)
X1 = X1/max_x1
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.2)


model = tf.keras.Sequential([
     tf.keras.layers.Dense(500, input_shape = (1,), activation='relu'),
     tf.keras.layers.Dense(500, activation='relu'),
     tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
  ])

model.compile(optimizer='sgd',
            loss='mean_squared_error',)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

X_predict = x[-num:]
X_prediction = X_predict/max(X_predict)
forecast = model.predict(X_prediction)*max_y2
forecast = forecast.reshape(num)


diff = 100*(forecast - Y2)/forecast
diff = np.abs(diff)
print("Ecart moyen entre la prediction et la réalité en pourcentage:", diff.mean())


plt.scatter(X_predict, forecast, c='b', linewidths=0.2)
plt.scatter(X_predict, Y2, c='r', alpha=0.5, linewidths=0.2)
plt.title("Réseau de neurones avec dates")
plt.legend(["predictions", "Reality"])
plt.xlabel("nb de minutes depuis le 1er juin")
plt.ylabel("valeur de fermeture")
plt.show()


#Le réseau de neurone ne fonctionne pas très bien car il doit prédire des valeurs très proches les unes des autres, il n'arrive qu'à détecter la tendance général
