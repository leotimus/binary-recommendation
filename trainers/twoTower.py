import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
from trainers.model_utils import getOptimizer
import time
from trainers.loadBinaryMovieLens import *

class twoTowerModel(tf.keras.Model):
	def __init__(self, embedDim, nbrItem, nbrUser):
		super().__init__(self)
		self.embedDim = embedDim
		self.nbrItem = nbrItem
		self.nbrUser = nbrUser
		#print(nbrItem)
		
		self.userTowerIn = tf.keras.layers.experimental.preprocessing.StringLookup(nbrUser)
		self.userTowerOut = tf.keras.layers.Embedding(nbrUser, embedDim)
		self.itemTowerIn = tf.keras.layers.experimental.preprocessing.StringLookup(nbrItem)
		self.itemTowerOut = tf.keras.layers.Embedding(nbrItem, embedDim)
		
		self.userTower = tf.keras.Sequential([self.userTowerIn, self.userTowerOut])
		self.itemTower = tf.keras.Sequential([self.itemTowerIn, self.itemTowerOut])
		
		self.outputLayer = tf.keras.Sequential(tf.keras.layers.Dense(1, input_shape = (2*embedDim,)))
	
	def call(self, info, training = False):
		"""userInput = self.userTowerIn(info["user_id"])
		usersCaracteristics = self.userTowerOut(userInput)
		itemInput = self.itemTowerIn(info["movie_id"])
		itemCaracteristics = self.itemTowerOut(itemInput)"""
		usersCaracteristics = self.userTower(info["user_id"])
		itemCaracteristics = self.itemTower(info["movie_id"])
		return self.outputLayer(tf.concat([usersCaracteristics, itemCaracteristics], -1))
	
	def train_step(self, info):
		with tf.GradientTape() as tape:
			pred = self(info, training = True)
			loss = self.compiled_loss(pred, info["rating"])
		
		#print(tape.watched_variables)
		gradients = tape.gradient(loss, self.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
		#metrics = {metric.name: metric.result() for metric in self.metrics}
		return {"loss":loss}
	
	def test_step(self, info):
		pred = self(info)
		loss = self.compiled_loss(pred, info["rating"])
		return {"loss":loss}

def splitTrainTest(data, ratio):
	dataSize = len(data)
	trainSetSize = int(dataSize*ratio)
	testSetSize = dataSize - trainSetSize
	shuffled = data.shuffle(dataSize, reshuffle_each_iteration=False)


	train = shuffled.take(trainSetSize)
	test = shuffled.skip(trainSetSize).take(testSetSize)
	return (train, test)
	


if __name__ == "__main__":
	data = movieLensData(1,2)
	#print(dict(data["ratings"]))
	ratings = tf.data.Dataset.from_tensor_slices(dict(data["ratings"]))
	trainSet, testSet = splitTrainTest(ratings, 0.8)
	model = twoTowerModel(32, data["nbrMovie"], data["nbrUser"])
	model.compile(optimizer = getOptimizer("Adam", 0.002), loss = "MSE")
	
	testSetCached = trainSet.batch(8000).cache()
	
	model.fit(testSetCached, epochs = 10)
	print("test")
	model.evaluate(testSet.batch(4000).cache(), return_dict=True)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
