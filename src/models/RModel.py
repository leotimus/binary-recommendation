import sys, os
from typing import Tuple

import tensorflow as tf
import matplotlib.pyplot as plt
from pandas import DataFrame
from tensorflow.keras.utils import model_to_dot
from sklearn.model_selection import train_test_split
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from tensorflow.python.distribute.distribute_lib import Strategy
from tensorflow.python.keras.models import Model

from src.datasource.DataStore import DataStore


class RModel:
  CUSTOMER_ID = 'CUSTOMER_ID'
  PRODUCT_ID = 'PRODUCT_ID'
  # METRICS = ['mse', 'mae', 'false_negatives', 'false_positives', 'true_negatives', 'true_positives', 'binary_accuracy']
  METRICS = ['mse', 'mae', 'binary_accuracy']

  def __init__(self, moduleName):
    self.modelName = moduleName

    self.modelStructurePath = 'export/{}/model.png'.format(self.modelName)
    self.trainResultPlotPath = 'export/{}/plot.png'.format(self.modelName)
    self.checkpointPath = 'checkpoints/{}/cp'.format(self.modelName)
    self.modelProducts = 'checkpoints/{}/modelData/products'.format(self.modelName)
    self.modelUsers = 'checkpoints/{}/modelData/users'.format(self.modelName)

    os.makedirs('export/{}'.format(self.modelName), exist_ok=True)
    os.makedirs('checkpoints/{}'.format(self.modelName), exist_ok=True)
    os.makedirs('checkpoints/{}/modelData'.format(self.modelName), exist_ok=True)

    self.numFactor: int = 32
    self._epochs: int = 10
    self._batchSize: int = 1024
    self._validationSteps: int = 20
    self._testSize: float = 0.2

    self._model: Model = None

    self._dataStore = DataStore()

  @property
  def testSize(self) -> float:
    return self._testSize

  @testSize.setter
  def testSize(self, value: float):
    self._testSize = value

  @property
  def validationSteps(self) -> int:
    return self._validationSteps

  @validationSteps.setter
  def validationSteps(self, value):
    self._validationSteps = value

  @property
  def epochs(self) -> int:
    return self._epochs

  @epochs.setter
  def epochs(self, value: int):
    self._epochs = value

  @property
  def batchSize(self) -> int:
    return self._batchSize

  @batchSize.setter
  def batchSize(self, value: int):
    self._batchSize = value

  @property
  def dataStore(self) -> DataStore:
    return self._dataStore

  @dataStore.setter
  def dataStore(self, value):
    pass

  @property
  def model(self) -> Model:
    return self._model

  @model.setter
  def model(self, value: Model):
    self._model = value

  def compileModel(self, distributedConfig, numUser: int, numItem: int, numFactor: int) -> Model:
    print('placeholder')
    return None

  def bootstrapDataset(self, df, negRatio=3., batchSize=128, shuffle=True) -> tf.data.Dataset:
    return None

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

  def train(self, path, rowLimit, metricDict:dict = {}, distributedConfig=None):
    if distributedConfig is None:
      strategy, trainDataset, testDataset, trainSplit = self.prepareToTrain(distributedConfig, path, rowLimit)
    else:
      strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
      with strategy.scope():
        strategy, trainDataset, testDataset, trainSplit = self.prepareToTrain(distributedConfig, path, rowLimit)

    try:
      model_to_dot(self.model, show_shapes=True).write(path=self.modelStructurePath, prog='dot', format='png')
    except:
      print('Could not plot model in dot format:', sys.exc_info()[0])
    self.model.summary()

    if distributedConfig is None:
      history = self.model.fit(trainDataset, validation_data=testDataset, epochs=self.epochs)
    else:
      history = self.model.fit(trainDataset,
                               validation_data=testDataset,
                               epochs=self.epochs,
                               steps_per_epoch=len(trainDataset) / self.epochs / self.getNumberOfWorkers(
                                 distributedConfig),
                               validation_steps=self.validationSteps)

    self.model.save(self.getModelSaveLocation(strategy))

    self.plot(history, metricDict)

    print("Evaluating trained model...")
    _, val = train_test_split(trainSplit, test_size=0.2)
    valDataset = self.bootstrapDataset(val, shuffle=False)

    evaluatedMetric = list(self.model.evaluate(valDataset, steps=self.validationSteps))

    self.clearSlaveTempDir(strategy)
    return {'result': 'completed', 'metrics': evaluatedMetric}

  def prepareToTrain(self, distributedConfig, path, rowLimit) -> Tuple[Strategy, DatasetV2, DatasetV2, list]:
    return None

  def readData(self, path, rowLimit) -> {int, int, DataFrame}:
    print('To be implemented on specific model')
    return None

  def getPredictDataFrame(self, customerId) -> DataFrame:
    return None

  def predictForUser(self, customerId, numberOfItem=5):
    return None

  def getPredictableUsers(self) -> list:
    return []

  def getPredictDataSet(self, customerId) -> DatasetV2:
    predictionDf = self.getPredictDataFrame(customerId)
    return self.bootstrapDataset(predictionDf, shuffle=False)

  def restoreFromLatestCheckPoint(self):
    self.model = tf.keras.models.load_model(self.checkpointPath)

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

  def readyToTrain(self):
    return True

  def getNumberOfWorkers(self, distributedConfig) -> int:
    return len(distributedConfig['cluster']['worker'])
