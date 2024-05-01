#Regréssion Polynomiale
#partie 1
#Q1.  Lisez le fichier "data.csv"
import pandas as pd #pour la manipulation et l'analyse de données
df = pd.read_csv('data.csv')
#afficher les 5 premier valeur du data 
df.head()
#Q2. dataframe des valeurs indépendantes (Volume et Wheight) et appelez cette variable X.
X = df[["Volume", "Weight"]]
#Q3.  lavaleur (CO2) dans une variable appelée y.
Y = df["CO2"]
#Q4. utiliser la méthode LinearRegression() pour créer un objet de régression linéaire.
from sklearn.linear_model import LinearRegression
model = LinearRegression() 
#Q5 . Entrener les valeur X et Y 
model.fit(X,Y) 
 
#Q6 . prédire combien de grammes de CO2 est dégagés pour chaque kilomètre parcouru pour une voiture équipée d’un moteur de 1,3 litre (1300 ml) et pesant 2300 kg .
car_features = [[1300, 2300]]  

predicted = model.predict(car_features)
#Q6. la valeur du coefficient du poids par rapport au CO2
model.coef_

# Partie II : Regréssion Polynomiale
 
#Q1 . librairies : numpy, matplotlib, sklearn.
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.datasets import make_regression

#Q2. dataset en important la fonction datasets.make_regression et utilisez la pour générer un problème de régression aléatoire de 100 exemples avec une seule variable avec y=x^2
x, y = make_regression(n_samples=100, n_features=1, noise=10) 
y = y**2
poly_features = PolynomialFeatures(degree=2, include_bias=False) 
x = poly_features.fit_transform(x).T

#Q3. Visualiser du données 
plt.scatter(x[:,0], y) 

#Q4. modèle avec SGDRegressor() sur 100 itérations avec un Learning rate de 0.0001.
from sklearn.linear_model import SGDRegressor
model = SGDRegressor(max_iter=100, eta0=0.0001) 

#Q5. 5- Entraîner le modèle
model.fit(x,y) 

#Q6 . la précision du modèle
model.score(x,y)

#Q7 . nouvelles prédictions avec la fonction predict() et tracer les résultats.
plt.scatter(x[:,0], y) 
#7
plt.scatter(x[:,0], model.predict(x), c='red', lw = 3)

#Q8 . Refaire le même travail en entraînant votre modèle sur 1000 itérations avec un Learning rate de 0.001.
from sklearn.linear_model import SGDRegressor

model = SGDRegressor(max_iter=1000, eta0=0.001) 
model.fit(x,y) 
plt.scatter(x[:,0], y) 
plt.scatter(x[:,0], model.predict(x), c='red', lw = 3)
model.score(x,y)