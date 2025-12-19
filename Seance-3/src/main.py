#coding:utf8
# Séance 3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Source des données : https://www.data.gouv.fr/datasets/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/
# Sources des données : production de M. Forriez, 2016-2023

#Question 4
with open("./data/resultats-elections-presidentielles-2022-1er-tour.csv","r") as fichier:
    contenu = pd.read_csv(fichier)
    df=pd.DataFrame(data=contenu)
    print(df)
    import pandas as pd
print("--------------------------------------------------------------------")

#Question 5
contenu.info()
colonnes_quantitatives = contenu.select_dtypes(include=["int64", "float64"])
import numpy as np

#Paramètres statistiques
moyennes = colonnes_quantitatives.mean().round(2)
print("Moyennes :\n", moyennes.tolist())
médianes = colonnes_quantitatives.median().round(2)
print("Médianes :", médianes.tolist())
modes = colonnes_quantitatives.mode().iloc[0].round(2)
print("Modes :", modes.tolist())
écarts_type = colonnes_quantitatives.std().round(2)
print("Ecarts-types :", écarts_type.tolist())
écarts_absolus_à_la_moyenne = (np.abs(colonnes_quantitatives - colonnes_quantitatives.mean())).mean().round(2)
print("Ecarts absolus à la moyenne : ",écarts_absolus_à_la_moyenne.tolist())
étendues = (colonnes_quantitatives.max() - colonnes_quantitatives.min()).round(2)
print("Etendues : ", étendues.tolist())
print("-------------------------------------------------")
#Changer le rendu visuel peut-être pour une meilleure clarté d'information


#Question 7:

#-Distance interquartile : 
Q1 = colonnes_quantitatives.quantile(0.25)
Q3 = colonnes_quantitatives.quantile(0.75)
Distance_interquartile = (Q3-Q1).round(2)
print("Distance interquartile : ",Distance_interquartile.tolist())

#-Distance interdécile : 
D1 = colonnes_quantitatives.quantile(0.1)
D9 = colonnes_quantitatives.quantile(0.9)
Distance_interdécile = (D9-D1).round(2)
print("Distance interdécile :",Distance_interdécile.tolist())
print("-----------------------------------------------------")


#Question 8 : 
#Créer dossier
chemin_img= "img"
import os
os.makedirs(chemin_img, exist_ok=True)

#Boîte à moustache
for col in colonnes_quantitatives.columns:
    plt.figure(figsize=(6,4))
    plt.boxplot(colonnes_quantitatives[col])
    plt.title(f"Boîte à moustache de {col}")
    plt.ylabel(col)
    plt.savefig(f"img/Boîte_à_moustache_{col}.png", bbox_inches='tight')
    plt.close()
print("--------------------------------------------------------")



#Question 9:
with open("./data/island-index.csv","r") as fichier:
    contenu = pd.read_csv(fichier)
    df=pd.DataFrame(data=contenu)
    print(df)
    import pandas as pd
print(df.columns)
print("--------------------------------------------------------------------")

#Algorithme
entre_0_10 = df.loc[df["Surface (km²)"] <=10]
print("Surface de moins de 10km² :",entre_0_10, len(entre_0_10))
print("- - - - - - - - - -")

entre_10_25 = df.loc[(df["Surface (km²)"] >10) & (df["Surface (km²)"]<=25)]
print("Surface entre 10 et 25 km²:",entre_10_25, len(entre_10_25))
print("- - - - - - - - - -")

entre_25_50 = df.loc[(df["Surface (km²)"] >25) & (df["Surface (km²)"]<=50)]
print("Surface entre 25 et 50 km²",entre_25_50, len(entre_25_50))
print("- - - - - - - - - -")

entre_50_100 = df.loc[(df["Surface (km²)"] >50) & (df["Surface (km²)"]<=100)]
print("Surface entre 50 et 100 km²: ",entre_50_100, len(entre_50_100))
print("- - - - - - - - - -")

entre_100_2500 = df.loc[(df["Surface (km²)"] >100) & (df["Surface (km²)"]<=2500)]
print("Surface entre 100 et 2500 km²: ",entre_100_2500, len(entre_100_2500))
print("- - - - - - - - - -")

entre_2500_5000 = df.loc[(df["Surface (km²)"] >2500) & (df["Surface (km²)"]<=5000)]
print("Surface entre 2500 et 5000 km²: ",entre_2500_5000, len(entre_2500_5000))
print("- - - - - - - - - -")

entre_5000_10000 = df.loc[(df["Surface (km²)"] >5000) & (df["Surface (km²)"]<=10000)]
print("Surface entre 5000 et 10000 km² :",entre_5000_10000, len(entre_5000_10000))
print("- - - - - - - - - -")

plus_de_10000 = df.loc[df["Surface (km²)"]>10000]
print("Surface de plus de 10 000km² :",plus_de_10000, len(plus_de_10000))
print("- - - - - - - - - -")


#Bonus :
bins = [0, 10, 25, 50, 100, 2500, 5000, 10000, float("inf")]
labels = [
    "≤ 10",
    "]10 ; 25]",
    "]25 ; 50]",
    "]50 ; 100]",
    "]100 ; 2500]",
    "]2500 ; 5000]",
    "]5000 ; 10000]",
    "> 10000"
]
df["Classes des surfaces (km²)"] = pd.cut(
    df["Surface (km²)"],
    bins=bins,
    labels=labels,
    include_lowest=True
)
df.to_csv(
    "Tableau_classes_îles.csv",
    index=False,
    encoding="utf-8"
)
df.to_excel(
    "Tableau_classes_îles.xlsx",
    index=False
)
print("-----------------------")



