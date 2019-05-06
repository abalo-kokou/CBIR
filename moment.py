# python3 moment.py obj63__350.png
#                    obj21__215.png

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Importation des packages
import time
import cv2
import pickle
from scipy.spatial import distance
import os.path
start_time = time.time()

# Lecture des données
pathDataset = "./base"
pathFichierTrain = "./train"

#Transformation de l'image en niveau de gris
def Gris(chemin):
    gray = cv2.imread(chemin,cv2.IMREAD_GRAYSCALE)
    _,im = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return im

#Normalisation de l'image
def normalisationImage(image):
    normImage = image/8
    normImage = normImage.astype('uint32')
    return normImage

#Calcul des moments de HU
def momentHu(image):
    moments = cv2.moments(image)
    huMoments = cv2.HuMoments(moments)
    return huMoments

#Calcul de distance euclidienne entre deux moments de HU   
def CalculDistance(moment1,moment2):
    distances = distance.euclidean(moment1,moment2)
    return distances

#Récupération du descripteur Unpickle
def unpickle_hist(fichier):   
    Unpkl=pickle.Unpickler(fichier)
    fic=(Unpkl.load())
    return fic
#Creation du fichier pour stockage des descripteurs
def pickle_hist(fichier,histogramme):   
    pkl=pickle.Pickler(fichier)
    pkl.dump(histogramme)

#Stockage des moments de HU 
def Apprentissage():
    f = open((pathFichierTrain+"/moment"+".txt"),'wb')
    listeImage = os.listdir(pathDataset)
    moment_obj = {}
    for image in listeImage:
        chemin = (pathDataset+"/"+image)
        gris = Gris(chemin)
        moment = momentHu(gris)
        moment_obj[image] = moment
        pickle_hist(f,moment_obj)
    f.close

#Chercher les plus proches voisins
def RessemblaceImage(cheminImageTest,k):
    listeDistance ={}
    #Calcul des moments de HU de l'image requete
    gris = Gris(cheminImageTest)
    moment_test = momentHu(gris)
    f = open((pathFichierTrain+"/moment"+".txt"),"rb")
    list_hist = unpickle_hist(f)
    for key, valeur in list_hist.items():
        d = CalculDistance(valeur,moment_test)
        listeDistance[d] = key
    listeDistances = sorted(listeDistance.items(), key=lambda t:t[0])
    f.close
    
    return(listeDistances[:k])   
    
def main():
    Apprentissage()
    print("-------------- DESCRIPTEUR DE FORME (Calcul de moment) ---------------")
    chemin = input("Entrer le nom de l'image requete : ")
    k = int(input("Indiquer le nombre de voisins recherchés (N) : "))
    listeImage = RessemblaceImage(chemin,k)
    chemin = chemin.split("/")
    classe = (chemin[len(chemin)-1]).split("_")
    classe = classe[0]
    
    print ("\nImages"+ "\t\t\t\t" +"Distances\n")
    nbTrouve = 0
    for i in range(len(listeImage)):
        print(listeImage[i][1]," \t\t ",listeImage[i][0])
        chaine = listeImage[i][1] 
        chaine = chaine.split("_")
        if(classe == chaine[0]):
            nbTrouve=1+nbTrouve
    
    #Evatualtion du systeme 
    print("\n--------------------------")    
    precision = nbTrouve/k  
    rappel = nbTrouve/55
    #rappel = nbTrouve/62
    fmesure = 2/((1/precision)+(1/rappel))
    print(" Rappel    = ", rappel)
    print(" Précision = ", precision)
    print(" F-mesure  = ", fmesure)

        
if __name__ == '__main__':
    main()

