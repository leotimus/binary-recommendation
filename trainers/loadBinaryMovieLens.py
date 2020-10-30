import pandas as pd

def movieLensData(ratedVal, unratedVal):
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
	for user in usersId:
		i += 1
		
		for movie in moviesId:
			if (user, movie) in realRat:
				unrated.append([user, movie, float(unratedVal)])
				
	
	unrated = pd.DataFrame(unrated, columns=["user_id", "movie_id", "rating"])
	ratings = ratings.append(unrated, ignore_index=True)
	
	return {"ratings":ratings, "nbrUser":nbrUser, "nbrMovie":nbrMovie, "realRat":realRat}
