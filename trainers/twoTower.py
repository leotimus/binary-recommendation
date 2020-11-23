import tensorflow as tf
import pandas as pd
from trainers.model_utils import getOptimizer
import time
from trainers.loadBinaryMovieLens import *
from trainers.topKMetrics import *
import tensorflow_recommenders as tfrs
import sys
from getpass import getpass

class TwoTowerModel(tf.keras.Model):
	def __init__(self, embedDim, nbrItem, nbrUser, userKey, itemKey, usersId, itemsId, eval_batch_size = 8000, loss = None):
		super().__init__(self)
		self.embedDim = embedDim
		self.nbrItem = nbrItem
		self.nbrUser = nbrUser
		self.userKey = userKey
		self.itemKey = itemKey
		self.streamingLayer = tfrs.layers.factorized_top_k.Streaming()
		self.eval_batch_size = eval_batch_size
		#print(nbrItem)
		
		self.userTowerIn = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary = usersId)
		self.userTowerOut = tf.keras.layers.Embedding(nbrUser+2, embedDim)
		self.itemTowerIn = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary = itemsId)
		self.itemTowerOut = tf.keras.layers.Embedding(nbrItem+2, embedDim)
		
		#self.userTower = tf.keras.Sequential([self.userTowerIn, self.userTowerOut, tf.keras.layers.Flatten(), tf.keras.layers.Dense(20)])
		#self.itemTower = tf.keras.Sequential([self.itemTowerIn, self.itemTowerOut, tf.keras.layers.Flatten(), tf.keras.layers.Dense(20)])
		self.userTower = tf.keras.Sequential([self.userTowerIn, self.userTowerOut])
		self.itemTower = tf.keras.Sequential([self.itemTowerIn, self.itemTowerOut])
		
		self.outputLayer = tf.keras.layers.Dot(axes=1)
		self.task = tfrs.tasks.Retrieval(loss = loss)
		
		#self.outputLayer = tf.keras.Sequential(tf.keras.layers.Dense(32, input_shape=(embedDim*2,), activation = "sigmoid"))
		#self.outputLayer.add(tf.keras.layers.Dense(1, activation = "sigmoid"))
	
	"""def call(self, info):
		usersCaracteristics = self.userTower(info[self.userKey])
		itemCaracteristics = self.itemTower(info[self.itemKey])
		#return self.outputLayer(tf.concat([usersCaracteristics, itemCaracteristics], -1))
		#return tf.keras.activations.sigmoid(self.outputLayer([usersCaracteristics, itemCaracteristics]))
		return self.outputLayer([usersCaracteristics, itemCaracteristics])"""
	
	def call(self, info):
		usersCaracteristics = self.userTower(info)
		return self.streamingLayer(usersCaracteristics)
	
	def setCandidates(self, items, k):
		self.streamingLayer = tfrs.layers.factorized_top_k.Streaming(k = k)
		self.streamingLayer.index(
						candidates = items.map(self.itemTower),
						identifiers = items
					)
	
	"""def getTopK(self, users, k):
		self.streamingLayer(
					query_embeddings = self.userTower(users), 
					k = k
				)"""
	
	def computeEmb(self, info):
		usersCaracteristics = self.userTower(info[self.userKey])
		itemCaracteristics = self.itemTower(info[self.itemKey])
		return (usersCaracteristics, itemCaracteristics)
	
	def train_step(self, info):
		with tf.GradientTape() as tape:
			#pred = self(info, training = True)
			#loss = self.compiled_loss(info[self.resKey], pred)
			usersCaracteristics, itemCaracteristics = self.computeEmb(info)
			loss = self.task(usersCaracteristics, itemCaracteristics, compute_metrics = False, training = True, candidate_ids = info[self.itemKey])
		
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
		loss = self.task(usersCaracteristics, itemCaracteristics, candidate_ids = info[self.itemKey])
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
	

def crossValidation(filenames, k, learningRate, optimiser, loss, epoch, embNum, batchSize):
	#Load the files for cross-validation.
	dataSets = []
	username = input("username:")
	psw = getpass()
	print("Loading files")
	for filename in filenames:
		dataSets.append(gfData(filename, username, psw))
	
	print("Loading done.")
	
	#getting all unique users id and materials id
	usersId = []
	matId = []
	datas = []
	for dataSet in dataSets:
		usersId.append(pd.Series(dataSet["usersId"]))
		matId.append(pd.Series(dataSet["materialsId"]))
		datas.append(dataSet["ratings"])
	usersId = pd.unique(pd.concat(usersId))
	matId = pd.unique(pd.concat(matId))
	dataSets = datas
	
	#cross-validation
	res = []
	for i in range(len(dataSets)):
		print("cross validation it: " + str(i) + "/" + str(len(dataSets)))
		#creating test set and training set
		testData = dataSets.pop(0)
		testSet = tf.data.Dataset.from_tensor_slices(dict(testData))
		trainSet = tf.data.Dataset.from_tensor_slices(dict(pd.concat(dataSets, ignore_index=True)))
		
		#creating model
		model = TwoTowerModel(embNum, len(matId), len(usersId), "CUSTOMER_ID", "MATERIAL", usersId, matId, eval_batch_size = batchSize, loss = loss)
		model.compile(optimizer = getOptimizer(optimiser, learningRate = learningRate))
		
		#preparing trainingSet
		trainSet = trainSet.shuffle(len(trainSet), reshuffle_each_iteration=False)
		trainSetCached = trainSet.batch(batchSize).cache()
		
		#training
		print("training")
		#model.fit(trainSetCached, epochs = epoch)
		
		#testing
		print("testing")
		#topk = topKRatings(k, model, usersId, matId, "two tower")
		model.setCandidates(tf.data.Dataset.from_tensor_slices(matId), k)
		topk = model.predict(tf.data.Dataset.from_tensor_slices(usersId))
		print(topk.numpy())
		res.append(topKMetrics(topk, [(str(int(i["CUSTOMER_ID"].numpy())), str(int(i["MATERIAL"].numpy()))) for i in testSet], usersId, matId))
		print(res[-1])
		
		#making ready for next it
		dataSets.append(testData)
	
	#computing average results
	averageMetrics = {}
	for metrics in res[0]:
		averageMetrics[metrics] = 0
		for itRes in res:
			averageMetrics[metrics] += itRes[metrics]
		
		averageMetrics[metrics] /= len(dataSets)
	
	return averageMetrics
	
	
		
		
	
	
	
	


if __name__ == "__main__":
	print(sys.argv)
	learningRate = 0.1
	optimiser = "Adagrad"
	splitRatio = 0.8
	loss = None
	filename = [r"(NEW)CleanDatasets\TT-NCF\2m(OG)\ds2_OG(2m)_timeDistributed.csv1.csv", r"(NEW)CleanDatasets\TT-NCF\2m(OG)\ds2_OG(2m)_timeDistributed.csv2.csv", r"(NEW)CleanDatasets\TT-NCF\2m(OG)\ds2_OG(2m)_timeDistributed.csv3.csv", r"(NEW)CleanDatasets\TT-NCF\2m(OG)\ds2_OG(2m)_timeDistributed.csv4.csv", r"(NEW)CleanDatasets\TT-NCF\2m(OG)\ds2_OG(2m)_timeDistributed.csv5.csv"]
	epoch = 3
	embNum = 32
	batchSize = 5000
	k = 10
	for i in range(len(sys.argv)):
		if sys.argv[i] == "data":
			filename = sys.argv[i+1]
		elif sys.argv[i] == "loss":
			loss = sys.argv[i+1]
		elif sys.argv[i] == "epoch":
			epoch = int(sys.argv[i+1])
		elif sys.argv[i] == "lrate":
			learningRate = float(sys.argv[i+1])
		elif sys.argv[i] == "ratio":
			splitRatio = float(sys.argv[i+1])
		elif sys.argv[i] == "k":
			k = float(sys.argv[i+1])
	
	res = crossValidation(filename, k, learningRate, optimiser, loss, epoch, embNum, batchSize)
	print(res)
	with open("../result/twoTowerResult", "a") as f:
		f.write("k: " + str(k) + ", learning rate: " + str(learningRate) + ", optimiser: " + optimiser + ", splitRatio: " + str(splitRatio) + ", loss: " + str(loss) + ", filename: " + filename + ", epoch: " + str(epoch) + "nbr embedings: " + str(embNum) + ", batchSize: " + str(batchSize) + "\n")
		f.write(str(res) + "\n")




	"""#data = movieLensData(1,0,0)
	data = gfData(filename)
	#print(dict(data["ratings"]))
	ratings = tf.data.Dataset.from_tensor_slices(dict(data["ratings"]))
	trainSet, testSet = splitTrainTest(ratings, splitRatio)
	#model = TwoTowerModel(embNum, data["nbrMovie"], data["nbrUser"], "user_id", "movie_id", "rating", data["usersId"], data["moviesId"], ratings)
	model = TwoTowerModel(embNum, data["nbrMaterial"], data["nbrUser"], "CUSTOMER_ID", "MATERIAL", "is_real", data["usersId"], data["materialsId"], ratings, eval_batch_size = batchSize, loss = loss)
	threshold = 0.5
	#model.compile(optimizer = getOptimizer("Adam",learningRate = 0.01), metrics=["MAE","MSE",tf.keras.metrics.BinaryAccuracy(threshold = threshold), tf.keras.metrics.TrueNegatives(threshold), tf.keras.metrics.TruePositives(threshold), tf.keras.metrics.FalseNegatives(threshold), tf.keras.metrics.FalsePositives(threshold)])
	model.compile(optimizer = getOptimizer(optimiser, learningRate = learningRate))
	
	trainSetCached = trainSet.batch(batchSize).cache()
	#trainSetCached = trainSet.batch(80000)
	#tf.keras.utils.plot_model(model, expand_nested = True)
	model.fit(trainSetCached, epochs = epoch)
	print("test")
	#res = model.evaluate(testSet.batch(batchSize).cache(), return_dict=True)
	#with open("../result/twoTowerResult", "a") as f:
	#	f.write("learning rate: " + str(learningRate) + ", optimiser: " + optimiser + ", splitRatio: " + str(splitRatio) + ", loss: " + str(loss) + ", filename: " + filename + ", epoch: " + str(epoch) + "nbr embedings: " + str(embNum) + ", batchSize: " + str(batchSize))
	#	f.write(str(res))
	#model.evaluate(testSet.batch(40000), return_dict=True)
	#raise Exception
	topk = topKRatings(10, model, data["usersId"], data["materialsId"], "two tower")
	print(topKMetrics(topk, [(str(int(i["CUSTOMER_ID"].numpy())), str(int(i["MATERIAL"].numpy()))) for i in testSet], data["usersId"], data["moviesId"]))"""
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
