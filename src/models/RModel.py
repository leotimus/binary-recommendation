import sys, os

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pandas import DataFrame
from tensorflow.keras.utils import model_to_dot
from sklearn.model_selection import train_test_split
from tensorflow.python.distribute.distribute_lib import Strategy
from tensorflow.python.keras.models import Model


class RModel:
  CUSTOMER_ID = 'CUSTOMER_ID'
  PRODUCT_ID = 'PRODUCT_ID'
  # METRICS = ['mse', 'mae', 'false_negatives', 'false_positives', 'true_negatives', 'true_positives', 'binary_accuracy']
  METRICS = ['mse', 'mae', 'binary_accuracy']

  def __init__(self, moduleName, batchSize=1024 * 16):
    self.modelName = moduleName

    self.modelStructurePath = 'export/{}/model.png'.format(self.modelName)
    self.trainResultPlotPath = 'export/{}/plot.png'.format(self.modelName)
    self.checkpointPath = 'checkpoints/{}/cp'.format(self.modelName)
    self.modelProducts = 'checkpoints/{}/modelData/products'.format(self.modelName)
    self.modelUsers = 'checkpoints/{}/modelData/users'.format(self.modelName)

    os.makedirs('export/{}'.format(self.modelName), exist_ok=True)
    os.makedirs('checkpoints/{}'.format(self.modelName), exist_ok=True)
    os.makedirs('checkpoints/{}/modelData'.format(self.modelName), exist_ok=True)

    self.numFactor = 32
    self.epochs = 10
    # self.batchSize = batchSize
    self.validationSteps = 20
    self._model = None

  @property
  def model(self):
    return self._model

  @model.setter
  def model(self, value):
    self._model = value

  def buildModel(self, numUser, numItem, numFactor) -> Model:
    print('placeholder')
    return None

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

  def plot(self, history, metrics):
    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.plot(history.history['loss'], label='loss')
    ax.plot(history.history['val_loss'], label='val_loss')
    for metric, label in metrics:
      ax.plot(history.history[metric], label=label)

    plt.ylim([0, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    fig.savefig(self.trainResultPlotPath)
    plt.close(fig)

  def train(self, path, rowLimit, metricDict, distributedConfig=None):
    numItem, numUser, transactionDf = self.readData(path, rowLimit)
    train, test = train_test_split(transactionDf, test_size=0.2)
    products = test.PRODUCT_ID.unique().tolist()
    users = test.CUSTOMER_ID.unique().tolist()
    pd.DataFrame({self.PRODUCT_ID: products}).to_pickle(self.modelProducts)
    pd.DataFrame({self.CUSTOMER_ID: users}).to_pickle(self.modelUsers)
    sampleSize = len(train)
    print(sampleSize, 'train examples')
    print(len(test), 'test examples')

    strategy = None
    if distributedConfig is None:
      batchSize = self.epochs
      self._model = self.buildModel(numUser, numItem, self.numFactor)
    else:
      numWorkers = len(distributedConfig['cluster']['worker'])
      batchSize = self.epochs * numWorkers
      strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
      with strategy.scope():
        self._model = self.buildModel(numUser, numItem, self.numFactor)

    try:
      model_to_dot(self._model, show_shapes=True).write(path=self.modelStructurePath, prog='dot', format='png')
    except:
      print('Could not plot model in dot format:', sys.exc_info()[0])
    self._model.summary()

    # train data set
    trainDataset = self.bootstrapDataset(train, batchSize=batchSize)
    # split validation dataset
    testDataset = self.bootstrapDataset(test, batchSize=batchSize, shuffle=False)

    if distributedConfig is None:
      history = self._model.fit(trainDataset, validation_data=testDataset, epochs=self.epochs)
    else:
      history = self._model.fit(trainDataset, validation_data=testDataset, epochs=self.epochs,
                                steps_per_epoch=len(trainDataset) / self.epochs / numWorkers, validation_steps=self.validationSteps)

    self._model.save(self.getModelSaveLocation(strategy))

    self.plot(history, metricDict)

    print("Evaluating trained model...")
    _, val = train_test_split(train, test_size=0.2)
    valDataset = self.bootstrapDataset(val, shuffle=False)

    evaluatedMetric = list(self._model.evaluate(valDataset, steps=20))

    self.clearSlaveTempDir(strategy)
    return {'result': 'completed', 'metrics': evaluatedMetric}

  def readData(self, path, rowLimit) -> {int, int, DataFrame}:
    df = pd.read_csv(path, nrows=rowLimit)
    transactionDf = df.drop(['MATERIAL', 'QUANTITY'], axis=1)
    numUser = transactionDf.CUSTOMER_ID.max() + 1
    numItem = transactionDf.PRODUCT_ID.max() + 1
    return numItem, numUser, transactionDf

  def getPredictDataSet(self, customerId):
    predictionDf = pd.read_pickle(self.modelProducts)
    predictionDf[self.CUSTOMER_ID] = customerId
    return self.bootstrapDataset(predictionDf, shuffle=False)

  def restoreFromLatestCheckPoint(self):
    self._model = tf.keras.models.load_model(self.checkpointPath)

  def predictForUser(self, customerId, numberOfItem=5):

    predictDataSet = self.getPredictDataSet(customerId)
    predictions = self._model.predict(predictDataSet)

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

  def getPredictableUsers(self):
    return pd.read_pickle(self.modelUsers).CUSTOMER_ID.tolist()

  def getModelSaveLocation(self, strategy: Strategy) -> str:
    if strategy is None or self.isMaster(strategy.cluster_resolver.task_type, strategy.cluster_resolver.task_id):
      return self.checkpointPath
    else:
      return self.getSlaveTempDir(strategy.cluster_resolver.task_id)

  def isMaster(self, taskType, taskId) -> bool:
    # If `task_type` is None, this may be operating as single worker, which works
    # effectively as chief.
    return taskType is None or taskType == 'chief' or (
      taskType == 'worker' and taskId == 0)

  def getSlaveTempDir(self, taskId):
    baseDirPath = 'workertemp_' + str(taskId)
    tempDir = os.path.join(self.checkpointPath, baseDirPath)
    tf.io.gfile.makedirs(tempDir)
    return tempDir

  def clearSlaveTempDir(self, strategy: Strategy):
    if strategy is not None and self.isMaster(strategy.cluster_resolver.task_type,
                                              strategy.cluster_resolver.task_id) is False:
      tf.io.gfile.rmtree(os.path.dirname(self.getSlaveTempDir(strategy.cluster_resolver.task_id)))
