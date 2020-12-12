import pandas as pd
from src.models.RModel import RModel

# map/link model saved files to checkpoints/NCFModel/cp/*
# map/link test.csv file to checkpoints/NCFModel/modelData/test2m.csv or change the trainData variable
class NCFModel(RModel):
  def __init__(self):
    super().__init__('NCFModel')
    self._productIds = []
    self._customerIds = []
    self.trainData = 'checkpoints/{}/modelData/test2m.csv'.format(self.modelName)
    self.loadTrainData()

  @property
  def customerIds(self) -> list:
      return self._customerIds

  @customerIds.setter
  def customerIds(self, ids:list):
      self._customerIds = ids

  @property
  def productIds(self) -> list:
      return self._productIds

  @productIds.setter
  def productIds(self, ids:list):
      self._productIds = ids

  def getPredictDataSet(self, customerId):
    return super().getPredictDataSet(customerId)

  def getPredictDataFrame(self, customerId):
    frame = pd.DataFrame({'PRODUCT_ID': self.productIds})
    frame['CUSTOMER_ID'] = customerId
    return frame

  def restoreFromLatestCheckPoint(self):
    super().restoreFromLatestCheckPoint()

  def predictForUser(self, customerId, numberOfItem=5):
    frame = self.getPredictDataFrame(customerId)
    predict = self.model.predict([frame.CUSTOMER_ID, frame.PRODUCT_ID], verbose=1)
    predictMap = dict(zip(self.productIds, predict.flatten()))
    predictMap = dict(filter(lambda elem: elem[1] < 1.0, predictMap.items()))
    sortedPredictMap = sorted(predictMap.items(), key=lambda x: x[1], reverse=True)[:numberOfItem]
    f = {}
    for k, v in sortedPredictMap:
      f[k] = '%.9f' % v
    return f

  # This model only support predict, train functions has not yet merged
  def readyToTrain(self):
    return False

  def getPredictableUsers(self):
    return self.customerIds

  def loadTrainData(self):
    #customer_id,normalized_customer_id,material,product_id,rating_type
    trainData = pd.read_csv(self.trainData)
    self.productIds = trainData.product_id.unique().tolist()
    self.customerIds = trainData.normalized_customer_id.unique().tolist()
