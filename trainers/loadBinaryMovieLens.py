import pandas as pd
import random as rd
from src.AAUfilename import *
from getpass import getpass


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
		realRat.add((ratings["user_id"][i], ratings["movie_id"][i]))
	
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
	
	return {"ratings":ratings, "nbrUser":nbrUser, "nbrMovie":nbrMovie, "realRat":realRat, "moviesId":moviesId, "usersId":usersId}

def gfData():
	cols = ['CUSTOMER_ID', 'MATERIAL', 'is_real']
	with smbc.open_file(getAAUfilename(r"CleanDatasets\no_0s\binary_MC_no0s_populated1000.csv"), mode="r", username=input("username: "), password=getpass()) as f:
		data = pd.read_csv(, names = cols, dtype = {"MATERIAL":str, "CUSTOMER_ID": str})
	data.drop(data.index[:1], inplace=True)
	data["is_real"] = data["is_real"].apply(lambda x: float(x))
	
	#ratings["MATERIAL"] = ratings["MATERIAL"].apply(lambda x : str(x))
	#ratings["CUSTOMER_ID"] = ratings["CUSTOMER_ID"].apply(lambda x : str(x))
	
	materialId = pd.unique(data["MATERIAL"])
	nbrMaterial = len(materialId)
	usersId = pd.unique(data["CUSTOMER_ID"])
	nbrUser = len(usersId)
	
	return {"ratings":data, "nbrUser":nbrUser, "nbrMaterial":nbrMaterial, "materialsId":materialId, "usersId":usersId}
	
	
	
	 





















































