import tensorflow as tf
import pandas as pd
from trainers.model_utils import getOptimizer
import time
from trainers.loadBinaryMovieLens import *
import tensorflow_recommenders as tfrs
import sys

class twoTowerModel(tf.keras.Model):
	def __init__(self, embedDim, nbrItem, nbrUser, userKey, itemKey, resKey, usersId, itemsId, itemsId_dataset, eval_batch_size = 8000, loss = None):
		super().__init__(self)
		self.embedDim = embedDim
		self.nbrItem = nbrItem
		self.nbrUser = nbrUser
		self.userKey = userKey
		self.itemKey = itemKey
		self.resKey = resKey
		#eval_batch_size = 80000
		#print(nbrItem)
		
		self.userTowerIn = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary = usersId)
		self.userTowerOut = tf.keras.layers.Embedding(nbrUser+2, embedDim)
		self.itemTowerIn = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary = itemsId)
		self.itemTowerOut = tf.keras.layers.Embedding(nbrItem+2, embedDim)
		
		#self.userTower = tf.keras.Sequential([self.userTowerIn, self.userTowerOut, tf.keras.layers.Flatten(), tf.keras.layers.Dense(20)])
		#self.itemTower = tf.keras.Sequential([self.itemTowerIn, self.itemTowerOut, tf.keras.layers.Flatten(), tf.keras.layers.Dense(20)])
		self.userTower = tf.keras.Sequential([self.userTowerIn, self.userTowerOut])
		self.itemTower = tf.keras.Sequential([self.itemTowerIn, self.itemTowerOut])
		
		#self.outputLayer = tf.keras.layers.Dot(axes=1)
		self.task = tfrs.tasks.Retrieval(
			loss = loss,
			metrics=tfrs.metrics.FactorizedTopK(
            		candidates=itemsId_dataset.map(lambda x: x[itemKey]).batch(eval_batch_size).map(self.itemTower)
			)
		)
		
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
		#return tf.keras.activations.sigmoid(self.outputLayer([usersCaracteristics, itemCaracteristics]))
		return None
	
	def computeEmb(self, info):
		usersCaracteristics = self.userTower(info[self.userKey])
		itemCaracteristics = self.itemTower(info[self.itemKey])
		return (usersCaracteristics, itemCaracteristics)
	
	def train_step(self, info):
		with tf.GradientTape() as tape:
			#pred = self(info, training = True)
			#loss = self.compiled_loss(info[self.resKey], pred)
			usersCaracteristics, itemCaracteristics = self.computeEmb(info)
			loss = self.task(usersCaracteristics, itemCaracteristics, training = True)
		
		#print(self.trainable_variables)
		gradients = tape.gradient(loss, self.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
		#self.compiled_metrics.update_state(info[self.resKey], pred)
		metrics = {m.name: m.result() for m in self.metrics}
		metrics["loss"] = loss
		return metrics
	
	def test_step(self, info):
		#pred = self(info)
		usersCaracteristics, itemCaracteristics = self.computeEmb(info)
		loss = self.task(usersCaracteristics, itemCaracteristics)
		#self.compiled_metrics.update_state(info[self.resKey], pred)
		metrics = {m.name: m.result() for m in self.metrics}
		metrics["loss"] = loss
		return metrics

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
	learningRate = 0.1
	optimiser = "Adam"
	splitRatio = 0.8
	loss = None
	filename = r"CleanDatasets\no_0s\binary_MC_global_no0s.csv"
	epoch = 10
	embNum = 32
	batchSize = 80000
	for i in range(len(sys.argv)):
		if sys.argv[i] == "data":
			filename = sys.argv[i+1]
		elif sys.argv[i] == "loss":
			loss = sys.argv[i+1]
		elif sys.argv[i] == "epoch":
			epoch = sys.argv[i+1]
		elif sys.argv[i] == "lrate":
			learningRate = sys.argv[i+1]
		elif sys.argv[i] == "ratio":
			splitRatio = sys.argv[i+1]
	
	#data = movieLensData(1,0,0)
	data = gfData(filename)
	#print(dict(data["ratings"]))
	data["ratings"] = tf.data.Dataset.from_tensor_slices(dict(data["ratings"]))
	ratings = data["ratings"]
	trainSet, testSet = splitTrainTest(ratings, splitRatio)
	#model = twoTowerModel(32, data["nbrMovie"], data["nbrUser"], "user_id", "movie_id", "rating", data["usersId"], data["moviesId"], data["ratings"])
	model = twoTowerModel(embNum, data["nbrMaterial"], data["nbrUser"], "CUSTOMER_ID", "MATERIAL", "is_real", data["usersId"], data["materialsId"], data["ratings"], eval_batch_size = batchSize, loss = loss)
	threshold = 0.5
	#model.compile(optimizer = getOptimizer("Adam",learningRate = 0.01), metrics=["MAE","MSE",tf.keras.metrics.BinaryAccuracy(threshold = threshold), tf.keras.metrics.TrueNegatives(threshold), tf.keras.metrics.TruePositives(threshold), tf.keras.metrics.FalseNegatives(threshold), tf.keras.metrics.FalsePositives(threshold)])
	model.compile(optimizer = getOptimizer(optimiser, learningRate = learningRate))
	
	testSetCached = trainSet.batch(batchSize).cache()
	#testSetCached = trainSet.batch(80000)
	#tf.keras.utils.plot_model(model, expand_nested = True)
	model.fit(testSetCached, epochs = epoch)
	print("test")
	model.evaluate(testSet.batch(batchSize).cache(), return_dict=True)
	#model.evaluate(testSet.batch(40000), return_dict=True)


	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
