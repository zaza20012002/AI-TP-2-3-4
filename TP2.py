import numpy as np #effectuer des calculs numériques
import pandas as pd #pour la manipulation et l'analyse de données
import matplotlib.pyplot as plt#créer des visualisations graphiques
import seaborn as sns #créer des visualisations statistiques attractives et informatives

# Importer les données du data 'Shoe prices'
shoes_dataset = pd.read_csv('Shoe prices.csv')
# Afficher des informations sur la data noms des colonnes, les types de données et les valeurs 
shoes_dataset.info()
# Afficher les noms des colonnes 
shoes_dataset.columns
# Vérifier les valeurs manquantes dans chaque colonne
shoes_dataset.isnull().sum()
shoes_dataset.describe()
# Afficher la forme (nombre de lignes et de colonnes) 
shoes_dataset.shape
# Afficher des donneés aléatoire de 4 lignes 
shoes_dataset.sample(4)
# Supprimer la colonne 'Modèle' 
shoes_dataset = shoes_dataset.drop('Modèle', axis=1)
shoes_dataset['Marque'].value_counts()
# Tracer un diagramme à barres montrant le décompte de chaque marque
shoes_dataset['Marque'].value_counts().plot(kind='bar', legend='false')
plt.title('Décompte des marques')
plt.xlabel('Noms des marques')
plt.ylabel('Décompte')
plt.show()
# Convertir toutes les valeurs de la colonne 'Type' en minuscules
shoes_dataset['Type'] = shoes_dataset['Type'].str.lower()
# Afficher du colonne 'Type'
shoes_dataset['Type'].value_counts()
# Tracer un diagramme à barres montrant la difference  de chaque type
shoes_dataset['Type'].value_counts().plot(kind='bar', legend='false', color='green')
plt.title('Décompte des types')
plt.xlabel('Noms des types')
plt.ylabel('Décompte')
plt.show()

# Définir une fonction pour catégoriser les types en 'sport' ou les laisser tels quels
def add_type(inpt):
    if inpt=='décontracté' or inpt=='mode' or inpt=='style de vie' or inpt=='diapositives' or inpt=='rétro':
        return inpt
    else:
        return 'sport'
        
# Appliquer la fonction add_type à la colonne 'Type'
shoes_dataset['Type'] = shoes_dataset['Type'].apply(add_type)
# Afficher les valeur unique dans la colonne 'Type' 
shoes_dataset['Type'].value_counts()
# Tracer un diagramme à barres montrant le décompte de chaque type après transformation
shoes_dataset['Type'].value_counts().plot(kind='bar', legend='false', color='green')
plt.title('Nouveaux décomptes de types')
plt.xlabel('Noms des types')
plt.ylabel('Décompte')
plt.show()

# Afficher le décompte de chaque valeur unique dans la colonne 'Genre'
shoes_dataset['Genre'].value_counts()

# Tracer un diagramme circulaire montrant la répartition des genres
shoes_dataset['Genre'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Répartition des genres')
plt.axis('equal')
plt.show()

# Supprimer 'US' des valeurs dans la colonne 'Taille' et la convertir en 'float'
shoes_dataset['Taille'] = shoes_dataset['Taille'].str.replace('US', '')
shoes_dataset['Taille'] = shoes_dataset['Taille'].astype(float)
# Afficher les 5 premières lignes  après modification de la colonne 'Taille'
shoes_dataset.head()
# Afficher des informations aprés modification de la colonne 'Taille'
shoes_dataset.info()
# Convertir toutes les valeurs dans la colonne 'Couleur' en minuscules
shoes_dataset['Couleur'] = shoes_dataset['Couleur'].str.lower()
# Afficher le décompte de chaque valeur unique dans la colonne 'Couleur'
shoes_dataset['Couleur'].value_counts()
# Tracer un diagramme à barres montrant le décompte de chaque couleur
shoes_dataset['Couleur'].value_counts().plot(kind='bar', legend='false', color='green')
plt.title('Décompte des couleurs')
plt.xlabel('Noms des couleurs')
plt.ylabel('Décompte')
plt.show()
#  fonction pour catégoriser les couleurs en 'autre' ou les laisser telles quelles
def add_Color(inpt):
    if inpt=='noir' or inpt=='blanc' or inpt=='gris' or inpt=='noir/blanc' or inpt=='rose':
        return inpt
    else:
        return 'autre'

# Appliquer la fonction add_Color à la colonne 'Couleur'
shoes_dataset['Couleur'] = shoes_dataset['Couleur'].apply(add_Color)
# Afficher le décompte de chaque valeur unique dans la colonne 'Couleur' après transformation
shoes_dataset['Couleur'].value_counts()
# Tracer un diagramme à barres montrant le décompte de chaque couleur après transformation
shoes_dataset['Couleur'].value_counts().plot(kind='bar', legend='false', color='green')
plt.title('Nouveaux décomptes de couleurs')
plt.xlabel('Noms des couleurs')
plt.ylabel('Décompte')
plt.show()

# Convertir toutes les valeurs dans la colonne 'Matériau' en minuscules
shoes_dataset['Matériau'] = shoes_dataset['Matériau'].str.lower()

# Afficher le décompte de chaque valeur unique dans la colonne 'Matériau'
shoes_dataset['Matériau'].value_counts()

# Tracer un diagramme à barres montrant le décompte de chaque matériau
shoes_dataset['Matériau'].value_counts().plot(kind='bar', legend='false', color='green')
plt.title('Décompte des matériaux')
plt.xlabel('Noms des matériaux')
plt.ylabel('Décompte')
plt.show()



# Supprimer le signe dollar des valeurs dans la colonne 'Prix (USD)' et la convertir en float
shoes_dataset['Prix (USD)'] = shoes_dataset['Prix (USD)'].str.replace('$', '').astype(float)

# Afficher les 5 premières lignes de la colonne 'Prix (USD)'
shoes_dataset.head(5)

# Afficher des informations sur la colonne 'Prix (USD)'
shoes_dataset.info()
#

# Séparer les caractéristiques (X) et la variable cible (y)
x = shoes_dataset.drop('Price (USD)', axis=1)
x 
#afficher les caractére du x et du y
y = shoes_dataset['Price (USD)']
x
