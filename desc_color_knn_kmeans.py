
# python3 desc_color_knn_kmeans.py -f obj63__350.png -M 16 -N 10 -C 5
# python3 desc_color_knn_kmeans.py -f obj63__350.png -M 16 -N 5 -C 5


# importation des packages necessaires 
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2


# Construction du parseur d'arguments et parser les arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--nomfichier", required = True,
	help = "Nom du fichier passe en argument")
ap.add_argument("-M", "--valeurM", required = True,
	help = "Valeur pour la reduction de l'histogramme")
ap.add_argument("-N", "--valeurN", required = True,
	help = "nombre d'images ayant les plus petites distances par rapport a notre image requete")
ap.add_argument("-C", "--valeurC", required = True,
	help = "nombre de clusters pour K-MEANS")
args = vars(ap.parse_args())

print ("\n\tImage de requete = " + args["nomfichier"])
print ("\tValeur de M = " + args["valeurM"])
print ("\tValeur de N = " + args["valeurN"])

# Initialise le dictionnaire d'index pour stocker le nom de l'image
# et les histogrammes correspondants et le dictionnaire d'images
# pour stocker les images elles-memes
index = {}
images = {}
nbTrouve = 0
classe_imgreq = args["nomfichier"].split("_")[0]
# Boucle dans le dossier des images pour les recuperer et traiter
for imagePath in glob.glob("base" + "/*.png"):
	# extract the image filename (assumed to be unique) and
	# load the image, updating the images dictionary
	filename = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)
	#images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# extract a 3D RGB color histogram from the im 20age,
	# using M bins per channel, normalize, and update
	# the index
	hist = cv2.calcHist([image], [0, 1, 2], None, [int(args["valeurM"]), int(args["valeurM"]), int(args["valeurM"])],
		[0, 256, 0, 256, 0, 256])
	#hists = cv2.normalize(hist).flatten()
	hist = cv2.normalize(hist, hist).flatten()
	index[filename] = hist


	# Initialisation du dictionnaire
results = {}
	
print ("\n\n KNN : " + args["valeurN"] + " plus proches voisins(" + args["valeurN"] + " images les plus similaires)\n")
print ("\nImages"+ "\t\t\t" +"Distances\n")

	# Boucle sur l'index
for (k, hist) in index.items():
		# compute the distance between the two histograms
		# using the method and update the results dictionary
        d = dist.cityblock(index[args["nomfichier"]], hist)
        results[k] = d

	# sort the results
results = sorted([(v, k) for (k, v) in results.items()])
	
	
'''
    calcul des k plus proches voisins et deduction de la classe a laquelle appartient l'image requete
'''
N=0
vote={}

for i in results :
    if N == int(args["valeurN"])+1 :
        break   
    if i[0] != 0.0 :
        image_requete=cv2.imread("base/"+args["nomfichier"])
        cv2.imshow("image requete",image_requete)
        voisin=cv2.imread("base/"+ str(i[1]))
        cv2.imshow("voisin"+str(N),voisin)
        print ("\""+str(i[1])+ "\"" + "\t\t" + str(i[0]))
        classe=str(str(i[1]).split("_")[0])
        #Nbre d'images retrouvees pour la classe de l'image requete
        if(classe == classe_imgreq):
            nbTrouve=nbTrouve + 1
        if classe in vote.keys() :
            vote[classe]=vote[classe]+1
        else :
            vote[classe]=1
    N=N+1


vote = sorted([(v, k) for (k, v) in vote.items()])

print ("\n Classe de l'image : " + vote[-1][1])
print ("\n Total images retrouvees pour la classe de l'image requete : " + str(nbTrouve))
print ("\n Nombre Total images retournees toutes classes confondues : " + str(N-1))
precision = nbTrouve/int(args["valeurN"]) 
rappel = nbTrouve/62
fmesure = 2/((1/precision)+(1/rappel))
print(" Rappel   = ", rappel)
print(" Précision = ", precision)
print(" F-mesure  = ", fmesure)



# K-means

print ("\n\n K-MEANS\n---------------------")

def initialiser_centres(hist_images, k):
	#returns k centroids from the initial points
    centroids_keys = list(hist_images.keys())
    np.random.shuffle(centroids_keys)  
    centroids=[]
    indice=0
    for centre in centroids_keys :
        if indice==k : 
            break
        centroids.append(centre)
        indice=indice+1
    return centroids

def kmeans(index, k) :
    centres=initialiser_centres(index, k)
    images_clusters={}
    for image in index.keys() :
        images_clusters[image]="1"
    for i in range(len(centres)) :
        images_clusters[centres[i]]=i+1
    for i in range(len(centres)) :
        print (centres[i])
    
    for (image, hist) in index.items() :      
        distances={}
        
        for centre in centres :                          
            distances[centre]=dist.cityblock(index[centre], hist)
        distances_ordonnees=sorted([(v, k) for (k, v) in distances.items()])
        
        images_clusters[image]=images_clusters[distances_ordonnees[0][1]]
        
    numero=1
    for (i,j) in images_clusters.items() :
        print (str(i) + "\t---->\tCluster_" + str(j))
        numero=numero+1

        
kmeans(index, int(args["valeurC"]))      
    
cv2.waitKey()
cv2.destroyAllWindows()

