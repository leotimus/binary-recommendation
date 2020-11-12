import tensorflow as tf
import pandas as pd
from trainers.model_utils import getOptimizer
import time
from trainers.loadBinaryMovieLens import *

class twoTowerModel(tf.keras.Model):
	def __init__(self, embedDim, nbrItem, nbrUser, userKey, itemKey, resKey, usersId, itemsId):
		super().__init__(self)
		self.embedDim = embedDim
		self.nbrItem = nbrItem
		self.nbrUser = nbrUser
		self.userKey = userKey
		self.itemKey = itemKey
		self.resKey = resKey
		#print(nbrItem)
		
		self.userTowerIn = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary = usersId)
		self.userTowerOut = tf.keras.layers.Embedding(nbrUser+2, embedDim)
		self.itemTowerIn = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary = itemsId)
		self.itemTowerOut = tf.keras.layers.Embedding(nbrItem+2, embedDim)
		
		self.userTower = tf.keras.Sequential([self.userTowerIn, self.userTowerOut, tf.keras.layers.Flatten(), tf.keras.layers.Dense(20)])
		self.itemTower = tf.keras.Sequential([self.itemTowerIn, self.itemTowerOut, tf.keras.layers.Flatten(), tf.keras.layers.Dense(20)])
		
		self.outputLayer = tf.keras.layers.Dot(axes=1)
		
		#self.outputLayer = tf.keras.Sequential(tf.keras.layers.Dense(32, input_shape=(embedDim*2,), activation = "sigmoid"))
		#self.outputLayer.add(tf.keras.layers.Dense(1, activation = "sigmoid"))
	
	def call(self, info, training = False):
		"""userInput = self.userTowerIn(info["user_id"])
		usersCaracteristics = self.userTowerOut(userInput)
		itemInput = self.itemTowerIn(info["movie_id"])
		itemCaracteristics = self.itemTowerOut(itemInput)"""
		usersCaracteristics = self.userTower(info[self.userKey])
		itemCaracteristics = self.itemTower(info[self.itemKey])
		#print(usersCaracteristics)
		#print(itemCaracteristics)
		#return self.outputLayer(tf.concat([usersCaracteristics, itemCaracteristics], -1))
		return tf.keras.activations.sigmoid(self.outputLayer([usersCaracteristics, itemCaracteristics]))
	
	def train_step(self, info):
		with tf.GradientTape() as tape:
			pred = self(info, training = True)
			loss = self.compiled_loss(info[self.resKey], pred)
		
		#print(self.trainable_variables)
		gradients = tape.gradient(loss, self.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
		self.compiled_metrics.update_state(info[self.resKey], pred)
		return {m.name: m.result() for m in self.metrics}
	
	def test_step(self, info):
		pred = self(info)
		self.compiled_metrics.update_state(info[self.resKey], pred)
		return {m.name: m.result() for m in self.metrics}

def splitTrainTest(data, ratio):
	dataSize = len(data)
	trainSetSize = int(dataSize*ratio)
	testSetSize = dataSize - trainSetSize
	shuffled = data.shuffle(dataSize, reshuffle_each_iteration=False)


	train = shuffled.take(trainSetSize)
	test = shuffled.skip(trainSetSize).take(testSetSize)
	return (train, test)
	

#def create_model


if __name__ == "__main__":
	#data = movieLensData(1,0,0.1)
	data = gfData()
	#print(dict(data["ratings"]))
	data["ratings"] = tf.data.Dataset.from_tensor_slices(dict(data["ratings"]))
	ratings = data["ratings"]
	trainSet, testSet = splitTrainTest(ratings, 0.8)
	#model = twoTowerModel(32, data["nbrMovie"], data["nbrUser"], "user_id", "movie_id", "rating", data["usersId"], data["moviesId"])
	model = twoTowerModel(32, data["nbrMaterial"], data["nbrUser"], "CUSTOMER_ID", "MATERIAL", "is_real", data["usersId"], data["materialsId"])
	threshold = 0.5
	model.compile(optimizer = getOptimizer("Adam",learningRate = 0.01), loss = "MSE", metrics=["MAE","MSE",tf.keras.metrics.BinaryAccuracy(threshold = threshold), tf.keras.metrics.TrueNegatives(threshold), tf.keras.metrics.TruePositives(threshold), tf.keras.metrics.FalseNegatives(threshold), tf.keras.metrics.FalsePositives(threshold)])
	
	testSetCached = trainSet.batch(80000).cache()
	#testSetCached = trainSet.batch(80000)
	#tf.keras.utils.plot_model(model, expand_nested = True)
	model.fit(testSetCached, epochs = 10)
	print("test")
	model.evaluate(testSet.batch(40000).cache(), return_dict=True)
	#model.evaluate(testSet.batch(40000), return_dict=True)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
