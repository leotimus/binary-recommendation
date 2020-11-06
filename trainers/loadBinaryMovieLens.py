import pandas as pd
import random as rd
from src.AAUfilename import *

def movieLensData(ratedVal, unratedVal, zeroProb):
	ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
	ratings = pd.read_csv('data/ml-100k/u.data', sep='\t', names=ratings_cols, encoding='latin-1')
	ratings = ratings.filter(items=["movie_id", "user_id", "rating"])
	
	
	ratings["movie_id"] = ratings["movie_id"].apply(lambda x : str(x))
	ratings["user_id"] = ratings["user_id"].apply(lambda x : str(x))
	ratings["rating"] = ratings["rating"].apply(lambda x: float(ratedVal))
	moviesId = pd.unique(ratings["movie_id"])
	nbrMovie = len(moviesId)
	usersId = pd.unique(ratings["user_id"])
	nbrUser = len(usersId)
	
	realRat = set()
	for i in range(len(ratings["movie_id"])):
		realRat.add((ratings["movie_id"][i], ratings["user_id"][i]))
	
	unrated = []
	i = 0
	bigNumber = 1000000
	for user in usersId:
		i += 1
		for movie in moviesId:
			if ((user, movie) not in realRat) and rd.randint(0,bigNumber) <= bigNumber*zeroProb:
				unrated.append([user, movie, float(unratedVal)])
				
	
	unrated = pd.DataFrame(unrated, columns=["user_id", "movie_id", "rating"])
	ratings = ratings.append(unrated, ignore_index=True)
	
	return {"ratings":ratings, "nbrUser":nbrUser, "nbrMovie":nbrMovie, "realRat":realRat}

def gfData(explicitVal, implicitVal, zeroProb):
	data = pd.read_csv(getAAUfilename("CleanDatasets/no_0s/binary_MC_global_no0s.csv"), dtype = {"MATERIAL":str, "CUSTOMER_ID": str})
	
	#ratings["MATERIAL"] = ratings["MATERIAL"].apply(lambda x : str(x))
	#ratings["CUSTOMER_ID"] = ratings["CUSTOMER_ID"].apply(lambda x : str(x))
	
	materialId = pd.unique(data["MATERIAL"])
	nbrMaterial = len(materialId)
	usersId = pd.unique(data["CUSTOMER_ID"])
	nbrUser = len(usersId)
	
	#add the expected output for the neural network
	data["is_real"] = [float(explicitVal) for i in range(len(data["MATERIAL"]))]
	
	#create a set containing the existing pairs
	explicit = set()
	for i in range(len(data["MATERIAL"])):
		explicit.add((data["MATERIAL"][i], data["CUSTOMER_ID"][i]))
	
	#populate with implicit values
	implicit = []
	i = 0
	bigNumber = 1000000
	for user in usersId:
		print("\r"+str(i)+"/"+str(nbrMaterial*nbrUser), end = "")
		for material in materialId:
			if ((user, material) not in explicit) and rd.randint(0,bigNumber) <= bigNumber*zeroProb:
				implicit.append([user, material, float(implicitVal)])
			i += 1
				
	
	implicit = pd.DataFrame(implicit, columns=["CUSTOMER_ID", "MATERIAL", "is_real"])
	data = data.append(implicit, ignore_index=True)
	
	return {"ratings":data, "nbrUser":nbrUser, "nbrMaterial":nbrMaterial, "realRat":explicit}
	
	
	
	 





















































