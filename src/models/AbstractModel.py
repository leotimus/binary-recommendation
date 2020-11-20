import sys
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import model_to_dot
from sklearn.model_selection import train_test_split


class AbstractModel:
  CUSTOMER_ID = 'CUSTOMER_ID'
  PRODUCT_ID = 'PRODUCT_ID'

  def __init__(self, name):
    self.name = name
    self.modelStructurePath = 'export/{}/model.png'.format(self.name)
    self.trainResultPlotPath = 'export/{}/plot.png'.format(self.name)
    self.checkpointPath = 'checkpoints/{}/cp'.format(self.name)
    self.num_factor = 32
    self.epochs = 10
    self.batchSize = 1024 * 16

  def getData(self, dataPath, rowLimit=None):
    df = pd.read_csv(dataPath, nrows=rowLimit)
    df = df.drop(['MATERIAL', 'QUANTITY'], axis=1)
    return df

  def buildModel(self, numUser, numItem, numFactor):
    print('placeholder')
    return None

  def bootstrapDataset(self, df, negRatio=3., batchSize=128, shuffle=True):
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

  def train(self, path, rowLimit, metricDict):
    transactionDf = self.getData(path, rowLimit)

    numUser = transactionDf.CUSTOMER_ID.max() + 1
    numItem = transactionDf.PRODUCT_ID.max() + 1


    model = self.buildModel(numUser, numItem, self.num_factor)
    try:
      model_to_dot(model, show_shapes=True).write(path=self.modelStructurePath, prog='dot', format='png')
    except:
      print('Could not plot model in dot format:', sys.exc_info()[0])
    model.summary()

    train, test = train_test_split(transactionDf, test_size=0.2)
    print(len(train), 'train examples')
    print(len(test), 'test examples')

    # Create a callback that saves the model's weights
    checkpointCallBack = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpointPath,
                                                            save_weights_only=True,
                                                            verbose=1)
    # train data set
    trainDataset = self.bootstrapDataset(train, self.epochs, self.batchSize)
    # split validation dataset
    testDataset = self.bootstrapDataset(test, self.epochs, self.batchSize, False)

    history = model.fit(trainDataset, validation_data=testDataset,
                                      epochs=self.epochs,
                                      callbacks=[checkpointCallBack])
    self.plot(history, metricDict)

    print("Evaluating trained model...")
    _, val = train_test_split(train, test_size=0.2)
    valDataset = self.bootstrapDataset(val, self.epochs, self.batchSize, False)

    returnMetrics = list(model.evaluate(valDataset))
    print("Accuracy: ", returnMetrics)

  def predict(self, customerId):

    model = self.buildModel()
    model.load_weights(self.checkpointPath)

    transactionDf = {}
    _, test = train_test_split(transactionDf, test_size=0.2)
    predict = self.bootstrapDataset(test, self.epochs, self.batchSize, False)

    predictions = model.predict(predict)

    i = 0
    extractFeatures = {}
    for features, label in predict.as_numpy_iterator():
      users = features['user']
      items = features['item']
      j = 0
      for u in users:
        if u == customerId:
          extractFeatures[items[j]] = predictions[i][0]
        j = j + 1
        i = i + 1
    print("predictions shape:", sorted(extractFeatures.items(), key=lambda x: x[1], reverse=True))
