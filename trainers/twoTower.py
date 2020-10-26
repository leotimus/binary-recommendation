import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
from trainers.model_utils import getOptimizer

class twoTowerModel(tfrs.Model):
	def __init__(self, embedDim, nbrItem, nbrUser, itemId, batchSize, loss):
		super().__init__(self)
		self.embedDim = embedDim
		self.nbrItem = nbrItem
		self.nbrUser = nbrUser
		self.batchSize = batchSize
		self.data = data
		#print(data)
		#print(nbrUser)
		
		self.userTower = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.IntegerLookup(nbrUser), tf.keras.layers.Embedding(nbrUser, embedDim)])
		self.itemTower = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.IntegerLookup(nbrItem), tf.keras.layers.Embedding(nbrItem, embedDim)])
		
		self.task = tfrs.tasks.Retrieval(
			loss = loss,
			metrics = tfrs.metrics.FactorizedTopK(candidates = itemId.batch(batchSize).map(self.itemTower), )
		)
	
	def compute_loss(self, data, training = False):
		print(data)
		usersCaracteristics = self.userTower(data["user_id"])
		itemsCaracteristics = self.itemTower(data["movie_id"])
		print(itemsCaracteristics)
		
		return self.task(usersCaracteristics, itemsCaracteristics)
	

def movieLensData():
	ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
	ratings = pd.read_csv('data/ml-100k/u.data', sep='\t', names=ratings_cols, encoding='latin-1')
	
	nbrMovie = len(pd.unique(ratings["movie_id"]))
	nbrUser = len(pd.unique(ratings["user_id"]))
	
	return {"ratings":ratings.filter(items=["movie_id", "user_id"]), "nbrUser":nbrUser, "nbrMovie":nbrMovie}

def splitTrainTest(data, ratio):
	dataSize = len(data)
	trainSetSize = int(dataSize*ratio)
	testSetSize = dataSize - trainSetSize
	shuffled = data.shuffle(dataSize, reshuffle_each_iteration=False)


	train = shuffled.take(trainSetSize)
	test = shuffled.skip(trainSetSize).take(testSetSize)
	return (train, test)
	


if __name__ == "__main__":
	data = movieLensData()
	#print(dict(data["ratings"]))
	ratings = tf.data.Dataset.from_tensor_slices(dict(data["ratings"]))
	itemsId = pd.unique(data["ratings"]["movie_id"])
	#print(itemsId)
	itemsId = tf.data.Dataset.from_tensor_slices(itemsId)
	#print(ratings)
	trainSet, testSet = splitTrainTest(ratings, 0.8)
	model = twoTowerModel(32, data["nbrMovie"], data["nbrUser"], itemsId, 128, None)#tf.keras.losses.MeanSquaredError())
	model.compile(optimizer = getOptimizer("SGD", 0.1))
	testSetCached = trainSet.batch(8192).cache()
	model.fit(testSetCached, epochs = 5)
	print(model.evaluate(testSetCached, return_dict=True))
	print(model.evaluate(testSet.batch(4096).cache(), return_dict=True))
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
