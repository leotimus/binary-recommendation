import numpy as np
# from tqdm import tqdm
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sys

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MSE
from tensorflow.keras.metrics import BinaryAccuracy

from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import Dense, Embedding, Dropout, BatchNormalization, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.utils import model_to_dot

MODEL_TO_DOT_PNG = 'export/model.png'
USER_FEATURE = 'CUSTOMER_ID'
PRODUCT_FEATURE = 'PRODUCT_ID'
CP_PATH = 'checkpoints/NeuMF02/cp'


def getData(rows=100000):
  dataPath = sys.argv[1] if len(sys.argv) >= 2 else ''
  nrows = int(sys.argv[2]) if len(sys.argv) >= 3 else rows
  df = pd.read_csv(dataPath, nrows=nrows)
  df = df.drop(['MATERIAL', 'QUANTITY'], axis=1);
  return df


def buildNeuralCollaborativeFilteringModel(numUser, numItem, numFactor):

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
  output = Dense(1, activation='sigmoid')(combine) #activation='sigmoid'

  inputs = [userId, itemId]
  model = Model(inputs, output, name='NeuMF')

  # model.compile(Adam(1e-3), loss=BinaryCrossentropy(), metrics=[BinaryAccuracy()])
  model.compile(Adam(1e-3), loss='mean_squared_error', metrics=['mse', 'mae', tf.keras.metrics.FalseNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.TruePositives(), BinaryAccuracy()])

  return model


def bootstrapDataset(df, negRatio=3., batchSize=128, shuffle=True):
  posDf = df[[USER_FEATURE, PRODUCT_FEATURE]].copy()
  negDf = df[[USER_FEATURE, PRODUCT_FEATURE]].sample(frac=negRatio, replace=True).copy()
  negDf.PRODUCT_ID = negDf.PRODUCT_ID.sample(frac=1.).values

  posDf['label'] = 1.
  negDf['label'] = 0.
  mergeDf = pd.concat([posDf, negDf]).sample(frac=1.)

  X = {
    "user": mergeDf[USER_FEATURE].values,
    "item": mergeDf[PRODUCT_FEATURE].values
  }
  Y = mergeDf.label.values

  ds = (tf.data.Dataset.from_tensor_slices((X, Y))
        .batch(batchSize))

  if shuffle:
    ds = ds.shuffle(buffer_size=len(df))

  return ds


def main():
  isTraining = False

  transactionDf = getData()

  numUser = transactionDf.CUSTOMER_ID.max() + 1
  numItem = transactionDf.PRODUCT_ID.max() + 1
  num_factor = 32
  epochs = 10
  batchSize = 1024 * 16

  model = buildNeuralCollaborativeFilteringModel(numUser, numItem, num_factor)
  try:
    model_to_dot(model, show_shapes=True).write(path=MODEL_TO_DOT_PNG, prog='dot', format='png')
  except:
    print('Could not plot model in dot format:', sys.exc_info()[0])

  model.summary()

  train, test = train_test_split(transactionDf, test_size=0.2)
  print(len(train), 'train examples')
  print(len(test), 'test examples')

  if isTraining:
    # Create a callback that saves the model's weights
    checkpointCallBack = tf.keras.callbacks.ModelCheckpoint(filepath=CP_PATH,
                                                            save_weights_only=True,
                                                            verbose=1)

    trainDataset = bootstrapDataset(train, epochs, batchSize)

    # split validation dataset
    testDataset = bootstrapDataset(test, epochs, batchSize, False)
    model.fit(trainDataset, validation_data=testDataset,
                            epochs=epochs,
                            callbacks=[checkpointCallBack])

    print("Evaluating trained model...")
    _, val = train_test_split(train, test_size=0.2)
    valDataset = bootstrapDataset(val, epochs, batchSize, False)
    loss, mse, mae, fn, fp, tn, tp, ba = model.evaluate(valDataset)
    print("Accuracy: ", [loss, mse, mae, fn, fp, tn, tp, ba])

  else:
    model.load_weights(CP_PATH)

    predict = bootstrapDataset(test, epochs, batchSize, False)

    predictions = model.predict(predict)
    i = 0
    extractFeatures = {}
    for features, label in predict.as_numpy_iterator():
      users = features['user']
      items = features['item']
      j = 0
      for u in users:
        if u == 190:
          extractFeatures[items[j]] = predictions[i][0]
        j = j + 1
        i = i + 1
    print("predictions shape:", sorted(extractFeatures.items(), key=lambda x: x[1], reverse=True))


if __name__ == "__main__":
  main()
