import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

df = pd.read_csv("DAT_MT_BCOUSD_M1_202005.csv")

df.columns = ['date', 'heure', 'open', 'high', 'low', 'close', 'nul']



diff = np.array(11)
while diff.mean()>10:
    Y = df['close'].values
    x = list(enumerate(Y))
    random.shuffle(x)
    x, y = zip(*x)
    X, Y = np.array(x), np.array(y)
    num = len(X)//10
    X = X[:, None]
    Y1 = Y[:-num]

    X1 = X[:-num]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.2)



    model = make_pipeline(PolynomialFeatures(12), Ridge())
    model.fit(X_train,y_train)
    model.score(X_test,y_test)

    X_predict = X[-num:]
    forecast = model.predict(X_predict)


    Y2 = Y[-num:]
    diff = 100*(forecast - Y2)/forecast
    diff = np.abs(diff)
print("Ecart moyen entre la prediction et la réalité en pourcentage:", diff.mean())

plt.scatter(X_predict, forecast)
plt.scatter(X_predict, Y2, c='r', alpha=0.7, linewidths=0.2)
plt.title("Regression polynomial degré 12")
plt.legend(["predictions", "Reality"])
plt.xlabel("nb de minutes depuis le 1er mai")
plt.ylabel("valeur de fermeture")
plt.show()

#La régression polynomiale fonctionne plutôt bien ici (cependant cela nécessite de faire des tests sur les degrés polynomiales)