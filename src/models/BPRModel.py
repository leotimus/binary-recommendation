from multiprocessing.pool import Pool
from typing import Tuple

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
from tensorflow.keras.layers import Embedding, Lambda, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.distribute.distribute_lib import Strategy

from src.models.RModel import RModel

class BPRModel(RModel):
  def __init__(self):
    super().__init__('BPRModel')
    self._trainDf: DataFrame = None
    self._productIds: list = []
    self._results: list = []

  @property
  def results(self) -> list:
    return self._results

  @results.setter
  def results(self, value: list):
    self._results = value

  @property
  def productIds(self) -> list:
    return self._productIds

  @productIds.setter
  def productIds(self, value: list):
    self._productIds = value

  @property
  def trainDf(self) -> DataFrame:
    return self._trainDf

  @trainDf.setter
  def trainDf(self, value: DataFrame):
    self._trainDf = value

  def compileModel(self, distributedConfig, numUser:int, numItem:int, numFactor:int) -> Tuple[Model, Strategy]:
    userInput = Input((1,), name='customerId_input')
    positiveItemInput = Input((1,), name='pProduct_input')
    negativeItemInput = Input((1,), name='nProduct_input')

    # One embedding layer is shared between positive and negative items
    itemEmbeddingLayer = Embedding(numItem, numFactor, name='item_embedding', input_length=1)

    positiveItemEmbedding = Flatten()(itemEmbeddingLayer(positiveItemInput))
    negativeItemEmbedding = Flatten()(itemEmbeddingLayer(negativeItemInput))

    userEmbedding = Embedding(numUser, numFactor, name='user_embedding', input_length=1)(userInput)
    userEmbedding = Flatten()(userEmbedding)

    tripletLoss = Lambda(self.bprTripletLoss, output_shape=self.outShape)([userEmbedding, positiveItemEmbedding, negativeItemEmbedding])

    # loss = merge([positiveItemEmbedding, negativeItemEmbedding, userEmbedding], mode=self.bprTripletLoss, name='loss', output_shape=(1,))

    self.model = Model(inputs=[userInput, positiveItemInput, negativeItemInput], outputs=tripletLoss)

    # manual loss function
    self.model.compile(loss=self.identityLoss, optimizer=Adam(1e-3))

    # self.model.compile(Adam(1e-3), loss='mean_squared_error', metrics=RModel.METRICS)

    return self.model, None

  def train(self, path, rowLimit, metricDict:dict = None, distributedConfig=None):
    self.batchSize = 64

    numItem, numUser, transactionDf = self.readData(path, rowLimit)

    # transactionDf.CUSTOMER_ID = transactionDf.CUSTOMER_ID.astype('category')
    # transactionDf.CUSTOMER_ID = transactionDf.CUSTOMER_ID.cat.codes
    # transactionDf.PRODUCT_ID = transactionDf.PRODUCT_ID.astype('category')
    # transactionDf.PRODUCT_ID = transactionDf.PRODUCT_ID.cat.codes

    self.trainDf, test = train_test_split(transactionDf, test_size=self.testSize)

    dfTriplets = pd.DataFrame(columns=['CUSTOMER_ID', 'pPRODUCT_ID', 'nPRODUCT_ID'])

    customerIds = self.trainDf.CUSTOMER_ID.unique().tolist()
    self.productIds = self.trainDf.PRODUCT_ID.unique().tolist()
    print(f'Having %s customers and %s products', str(len(customerIds)), str(len(self._productIds)))

    with Pool(5) as p:
      self.results.extend(p.map(self.extractPositivesNegatives, customerIds))

    for r in self.results:
      dfTriplets = dfTriplets.append(r, ignore_index=True)

    X = {
      'customerId_input': tf.convert_to_tensor(np.asarray(dfTriplets.CUSTOMER_ID).astype(np.float32)),
      'pProduct_input': tf.convert_to_tensor(np.asarray(dfTriplets.pPRODUCT_ID).astype(np.float32)),
      'nProduct_input': tf.convert_to_tensor(np.asarray(dfTriplets.nPRODUCT_ID).astype(np.float32))
    }

    # self.model = self.buildModel(numUser, numItem, self.numFactor)
    self.model = self.compileModel(max(customerIds) + 1, max(self.productIds) + 1, self.numFactor)

    self.model.fit(X, tf.ones(dfTriplets.shape[0]), batch_size=self.batchSize, epochs=self.epochs)

  def extractPositivesNegatives(self, customerId) -> list:
    entries = []
    print(f'processing customerId %s', customerId)
    existingProductIds = self._trainDf[self._trainDf.CUSTOMER_ID == customerId].PRODUCT_ID.tolist()
    for existingProductId in existingProductIds:
      for productId in self._productIds:
        if productId not in existingProductIds:
          entries.append({'CUSTOMER_ID': customerId, 'pPRODUCT_ID': existingProductId, 'nPRODUCT_ID': productId})
    return entries

  def outShape(self, shapes):
    return shapes[0]

  @tf.function
  def identityLoss(self, _, y_pred):
    return tf.math.reduce_mean(y_pred)

  @tf.function
  def bprTripletLoss(self, X: dict):
    """
    Calculate triplet loss - as higher the difference between positive interactions
    and negative interactions as better

    :param X: X contains the user input, positive item input, negative item input
    :return:
    """
    userLatent, positiveItemLatent, negativeItemLatent = X

    positiveInteractions = tf.math.reduce_sum(tf.math.multiply(userLatent, positiveItemLatent), axis=-1,
                                              keepdims=True)
    negativeInteractions = tf.math.reduce_sum(tf.math.multiply(userLatent, negativeItemLatent), axis=-1,
                                              keepdims=True)

    return tf.math.subtract(tf.constant(1.0), tf.sigmoid(tf.math.subtract(positiveInteractions, negativeInteractions)))
