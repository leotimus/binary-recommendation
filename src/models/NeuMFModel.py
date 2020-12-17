from typing import Tuple

import tensorflow as tf
from pandas import DataFrame
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Concatenate, Dense, Embedding, Dropout, BatchNormalization, Dot
from tensorflow.keras.models import Model
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from tensorflow.python.distribute.distribute_lib import Strategy

from src.models.RModel import RModel


class NeuMFModel(RModel):

  def __init__(self):
    super().__init__('NeuMFModel')

  def readData(self, path, rowLimit) -> {int, int, DataFrame}:
    file = self.dataStore.openFile(path=path, mode='r')
    df = pd.read_csv(file, nrows=rowLimit)
    transactionDf = df.drop(['MATERIAL', 'QUANTITY'], axis=1)
    numUser = transactionDf.CUSTOMER_ID.max() + 1
    numItem = transactionDf.PRODUCT_ID.max() + 1
    return numItem, numUser, transactionDf

  def prepareToTrain(self, distributedConfig, path, rowLimit) -> Tuple[DatasetV2, DatasetV2, list]:
    numItem, numUser, transactionDf = self.readData(path, rowLimit)

    trainSplit, testSplit = train_test_split(transactionDf, test_size=self.testSize)

    products = testSplit.PRODUCT_ID.unique().tolist()
    users = testSplit.CUSTOMER_ID.unique().tolist()

    pd.DataFrame({self.PRODUCT_ID: products}).to_pickle(self.modelProducts)
    pd.DataFrame({self.CUSTOMER_ID: users}).to_pickle(self.modelUsers)

    print(len(trainSplit), 'train examples')
    print(len(testSplit), 'testSplit examples')

    if distributedConfig is None:
      trainDataset = self.bootstrapDataset(trainSplit)
      testDataset = self.bootstrapDataset(testSplit, shuffle=False)
    else:
      trainDataset = self.bootstrapDataset(trainSplit, batchSize=self.batchSize)
      testDataset = self.bootstrapDataset(testSplit, batchSize=self.batchSize, shuffle=False)

    self.model = self.compileModel(distributedConfig, numUser, numItem, self.numFactor)
    return trainDataset, testDataset, trainSplit

  def compileModel(self, distributedConfig, numUser:int, numItem:int, numFactor:int) -> Model:
    userId = Input(shape=(), name='user')
    itemId = Input(shape=(), name='item')

    # MLP Embeddings
    userMLPEmbedding = Embedding(numUser, numFactor)(userId)
    itemMLPEmbedding = Embedding(numItem, numFactor)(itemId)

    # MF Embeddings
    userMFEmbedding = Embedding(numUser, numFactor)(userId)
    itemMFEmbedding = Embedding(numItem, numFactor)(itemId)

    # MLP Layers
    concatEmbedding = Concatenate()([userMLPEmbedding, itemMLPEmbedding])
    dropout = Dropout(0.2)(concatEmbedding)

    hidden1 = Dense(numFactor, activation='relu')(dropout)
    hidden1BN = BatchNormalization(name='bn1')(hidden1)
    dropout1 = Dropout(0.2)(hidden1BN)

    hidden2 = Dense(numFactor // 2, activation='relu')(dropout1)
    hidden2BN = BatchNormalization(name='bn2')(hidden2)
    dropout2 = Dropout(0.2)(hidden2BN)

    # Prediction from both layers
    hidden3MLP = Dense(numFactor // 4, activation='relu')(dropout2)
    predMF = Dot(axes=1)([userMFEmbedding, itemMFEmbedding])
    combine = Concatenate()([hidden3MLP, predMF])

    # Final prediction
    output = Dense(1, activation='sigmoid')(combine)

    inputs = [userId, itemId]

    if distributedConfig is None:
      self.model = Model(inputs, output, name=self.modelName)
      self.model.compile(Adam(1e-3),
                         loss='mean_squared_error',
                         metrics=RModel.METRICS)
    else:
      numWorkers = self.getNumberOfWorkers(distributedConfig)
      self.batchSize = self.epochs * numWorkers
      self.model = Model(inputs, output, name=self.modelName)
      self.model.compile(Adam(1e-3),
                           loss='mean_squared_error',
                           metrics=RModel.METRICS)

    return self.model

  def bootstrapDataset(self, df, negRatio=3., batchSize=128, shuffle=True) -> tf.data.Dataset:
    posDf = df[[self.CUSTOMER_ID, self.PRODUCT_ID]].copy()
    negDf = df[[self.CUSTOMER_ID, self.PRODUCT_ID]].sample(frac=negRatio, replace=True).copy()
    negDf.PRODUCT_ID = negDf.PRODUCT_ID.sample(frac=1.).values

    posDf['label'] = 1.
    negDf['label'] = 0.
    mergeDf = pd.concat([posDf, negDf]).sample(frac=1.)

    X = {
      "user": mergeDf[self.CUSTOMER_ID].values,
      "item": mergeDf[self.PRODUCT_ID].values
    }
    Y = mergeDf.label.values

    ds = (tf.data.Dataset.from_tensor_slices((X, Y))
          .batch(batchSize))

    if shuffle:
      ds = ds.shuffle(buffer_size=len(df))

    return ds

  def getPredictableUsers(self) -> list:
    return pd.read_pickle(self.modelUsers).CUSTOMER_ID.tolist()

  def getPredictDataFrame(self, customerId) -> DataFrame:
    predictionDf = pd.read_pickle(self.modelProducts)
    predictionDf[self.CUSTOMER_ID] = customerId
    return predictionDf

  def predictForUser(self, customerId, numberOfItem=5):

    predictDataSet = self.getPredictDataSet(customerId)
    predictions = self.model.predict(predictDataSet)

    i = 0
    extractFeatures = {}
    for features, label in predictDataSet.as_numpy_iterator():
      users = features['user']
      items = features['item']
      j = 0
      for u in users:
        if u == customerId:
          extractFeatures[str(items[j])] = str(predictions[i][0])
        j = j + 1
        i = i + 1

    return sorted(extractFeatures.items(), key=lambda x: x[1], reverse=True)[:numberOfItem]
