import tensorflow as tf
import tensorflow_recommenders as tfrs

class twoTowerModel(tfrs.Model):
	def __init__(self, embedDim, nbrItem, nbrUser):
		super().__init__(self)
		self.embedDim = embedDim
		self.nbrItem = nbrItem
		self.nbrUser = nbrUser
		
		self.userClassificator = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.StringLookup(max_tokens = nbrUser), tf.keras.layers.Embedding(nbrUser, embedDim))
		self.itemClassificator = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.StringLookup(max_tokens = nbrItem), tf.keras.layers.Embedding(nbrItem, embedDim))
		
		self.task = tfrs.tasks.Retrieval(
			metrics = FactorizedTopK()
		)
		
