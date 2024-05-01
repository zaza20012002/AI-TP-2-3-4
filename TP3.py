#Partie 1 :
#Q.1/ Fonction de cout pour le cas de f(x) = ax+b.
#J(w,b) = 1/2m Sum ((wx+b - y)^2)
#Q.2/ fonction compute_error qui calcule le fonction de cout qui sont l'emsemble d'error
#
import numpy as np
import matplotlib.pyplot as plt

def compute_error(b,w,points):
    totalerror=0
    for i in range(0,len(points)):    # Parcours de tous les points de données
        x=points[i,0]
        y=points[i,1]

        totalerror+=(y-(w*x+b))**2
    return totalerror/(2*float(len(points)))

#Q.3/ fonction nommée calcule_gradient qui calcule les nouveaux w et b 
#mises a jour en utilisant les dérivées partielles
#
def calcule_gradient(b_current,w_current,points,learnrate):
    b_gradient=0
    w_gradient=0
    N=float(len(points))

    for i in range(0,len(points)):
        x=points[i,0]
        y=points[i,1]

        #
        b_gradient+=(1/N)*((w_current*x+b_current)-y)#
        w_gradient+=(1/N)*x*((w_current*x+b_current)-y)
    # Calcul des gradients pour b et w
    new_b=b_current-(learnrate*b_gradient)
    new_w=w_current-(learnrate*w_gradient)
    # Retourne les nouvelles valeurs de b et w
    return [new_b,new_w]

#Q.4/ la fonction nommée GD_runner qui calcule les nauveux b ,w est qui affiche pour chaque itération
# la valeur du fonction de cout
# la valeur du coefficient w
#lavaleur du coefficient b
def gradient_descent_runner(points,starting_b,starting_w,learning_rate,num_iteration):
    b=starting_b
    w=starting_w
    cost_history = np.zeros(num_iteration)
    for i in range(num_iteration):
        b,w=step_gradent(b,w,np.array(points),learning_rate)
        print("Iteration Numero :",i)
        print("Fonction du cout = ", compute_error_for_line_given_point(b,w,np.array(points)))
        cost_history[i] = compute_error_for_line_given_point(b,w,np.array(points))
        print("w=", w)
        print("b=", b)
    plt.plot(range(num_iteration), cost_history)
    plt.show()
    return [b,w]




#PARTIE 2:
#les biblio numpy , matplotlib, sklearn.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor

#créer une dataset qui gérér un probléme de regression aléatoir de 100 exemples avec une seule variable
x,y=make_regression(n_samples=100, n_features=1,noise=10)

#la visualisation du donneé 
plt.scatter(x,y)

#SGDRegressor() sur 100 itérations avec un learning rate de 0.0001
model=SGDRegressor(max_iter=100,eta0=0.0001)

#Entrainer le modele
model.fit(x,y)
#tracer les résultats avec la fonction plt.plot()
plt.scatter(x, y) 
plt.plot(x, model.predict(x), c='red', lw = 3)

#Refaire le même travail en entraînant votre modèle sur 1000 itérations avec un Learning rate de 0.001
model=SGDRegressor(max_iter=1000,eta0=0.001)
model.fit(x,y)
plt.plot(x, model.predict(x), c='red', lw = 3)


# en remarque que le 2 module contient moin d'error que le premier 