# python3 cooccurence.py obj63__350.png

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importation des packages
import cv2
import os.path
from skimage.feature import greycomatrix
from skimage.feature.texture import greycoprops
import numpy as np
import pickle

# Lecture des données
pathDataset = "./base"
pathFichierTrain = "./train"

#Transformation de l'image en niveau de gris
def Gris(chemin):
    image = cv2.imread(chemin)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = normalisationImage(gray)
    return gray

#Normalisation de l'image
def normalisationImage(image):
    normImage = image/8
    normImage = normImage.astype('uint32')
    return normImage

    
#Histogramme de l'image à niveau de gris
def Histogramme(mat):
    histogramme = cv2.calcHist([mat], [0], None, [8, 8, 8],[0, 256])
    return histogramme
    
#Calcul de la distance 
def CalculDistance(image1,image2):
    d = (np.linalg.norm((image1-image2))/5)
    return d

#Matrice de cooccurence
def MatCooccurence(image_gris):
    matCo = greycomatrix(image_gris, [5], [0,np.pi/2,np.pi/4,(np.pi*3)/4], 256, symmetric=True, normed=True)
    return matCo
    print(matCo)
# Fonction de calcul des parametre de la cooccurence
def ParamCooccurence(HistomatCoo):
    #Calcul de l'energie
    energie = greycoprops(HistomatCoo,'energy')
    contraste = greycoprops(HistomatCoo,'contrast')
    dissimilarite = greycoprops(HistomatCoo,'dissimilarity')
    homogeneite = greycoprops(HistomatCoo,'homogeneity')
    correlation = greycoprops(HistomatCoo,'correlation')
    return energie, contraste, dissimilarite,homogeneite,correlation

   
    #Récupération du descripteur Unpickle
def unpickle_hist(fichier):   
    Unpkl=pickle.Unpickler(fichier)
    fic=(Unpkl.load())
    return fic
#Creation du fichier pour stockage des descripteurs
def pickle_hist(fichier,histogramme):   
    pkl=pickle.Pickler(fichier)
    pkl.dump(histogramme)

#Fonction d'apprentissage
def Apprentissage():
    f = open((pathFichierTrain+"/cooccurence"+".txt"),'wb')
    listeImage = os.listdir(pathDataset)
    paramCooccure = {}
    for image in listeImage:
        param = np.zeros(5)
        greyImage = Gris(pathDataset+"/"+image)
        MatCoo = MatCooccurence(greyImage)
        energie,contraste,dissimilarite,homogeneite,correlation = ParamCooccurence(MatCoo)
        energie = energie[0][0]
        param[0] = energie
        contraste = contraste[0][0]
        param[1] = contraste
        dissimilarite = dissimilarite[0][0]
        param[2] = dissimilarite
        homogeneite = homogeneite[0][0]
        param[3] = homogeneite
        correlation = correlation[0][0]
        param[4] = correlation
        paramCooccure[image] = param
        
    pickle_hist(f,paramCooccure)
    f.close
 
    
#Recherche des ressemblances   
def Ressemblance(chemin,k):
    listeDistance ={}
    param = np.zeros(5)
    image_gris = Gris(chemin)
    MatCoo = MatCooccurence(image_gris)
    energie,contraste,dissimilarite,homogeneite,correlation = ParamCooccurence(MatCoo)
    energie = energie[0][0]
    param[0] = energie
    contraste = contraste[0][0]
    param[1] = contraste
    dissimilarite = dissimilarite[0][0]
    param[2] = dissimilarite
    homogeneite = homogeneite[0][0]
    param[3] = homogeneite
    correlation = correlation[0][0]
    param[4] = correlation
    f = open((pathFichierTrain+"/cooccurence"+".txt"),"rb")
    list_hist = unpickle_hist(f)
    for key, valeur in list_hist.items():
        #valeur = np.asanyarray(valeur)
        listeDistance[CalculDistance(valeur,param)] = key
    listeDistances = sorted(listeDistance.items(), key=lambda t: t[0])
    f.close
    return(listeDistances[:k])

def main():
    Apprentissage()
    print("--------- MATRICE DE COOCURRENCE -------------")
    chemin = input("Entrer le nom de l'image requete : ")
    k = int(input("Indiquer le nombre de voisins recherchés (N) : "))
    
    listeImage = Ressemblance(chemin,k)
    chemin = chemin.split("/")
    classe = (chemin[len(chemin)-1]).split("_")
    classe = classe[0]
    
    print ("\nImages"+ "\t\t\t\t" +"Distances\n")
    nbTrouve = 0
    for i in range(len(listeImage)):
        print(listeImage[i][1]," \t\t",listeImage[i][0])
        chaine = listeImage[i][1] 
        chaine = chaine.split("_")
        if(classe == chaine[0]):
            nbTrouve=1+nbTrouve
    
    #Evaluation de performance
    print("\n--------------------------")      
    precision = nbTrouve/k  
    rappel = nbTrouve/62
    fmesure = 2/((1/precision)+(1/rappel))
    print(" Rappel    = ", rappel)
    print(" Précision = ", precision)
    print(" F-mesure  = ", fmesure)
    
if __name__ == '__main__':
    main()     
