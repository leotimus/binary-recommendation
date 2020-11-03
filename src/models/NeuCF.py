import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.prepare.DataFetcher import downloadData

tqdm.pandas()
np.set_printoptions(5, )
assert int(tf.__version__[0]) >= 2, "tensorflow 2.x should be installed"

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy

from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.models import Model


def getData(rows=200000):
  downloadData()
  df = pd.read_csv('prepare/lastfm_play.csv', nrows=rows)
  return df


def buildNeuralCollaborativeFilteringModel(numUser, numItem, numFactor):
  userId = Input(shape=(), name='user')
  itemId = Input(shape=(), name='item')

  userEmbedding = Embedding(numUser, numFactor)(userId)
  itemEmbedding = Embedding(numItem, numFactor)(itemId)

  concatEmbedding = Concatenate()([userEmbedding, itemEmbedding])

  hidden1 = Dense(numFactor, activation='relu')(concatEmbedding)
  hidden2 = Dense(numFactor // 2, activation='relu')(hidden1)
  hidden3 = Dense(numFactor // 4, activation='relu')(hidden2)

  output = Dense(1, activation='sigmoid')(hidden3)

  inputs = [userId, itemId]
  model = Model(inputs, output, name='NCF')

  model.compile(Adam(1e-3), loss=BinaryCrossentropy(), metrics=[BinaryAccuracy()])

  return model


def bootstrapDataset(df, negRatio=3., batchSize=128, shuffle=True):
  posDf = df[['user_id', 'artist_id']].copy()
  negDf = df[['user_id', 'artist_id']].sample(frac=negRatio, replace=True).copy()
  negDf.artist_id = negDf.artist_id.sample(frac=1.).values

  posDf['label'] = 1.
  negDf['label'] = 0.
  mergeDf = pd.concat([posDf, negDf]).sample(frac=1.)

  X = {
    "user": mergeDf['user_id'].values,
    "item": mergeDf['artist_id'].values
  }
  Y = mergeDf.label.values

  ds = (tf.data.Dataset.from_tensor_slices((X, Y))
        .batch(batchSize))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(df))
  return ds


def main():
  checkpointPath = "prepare/checkpoints/NeuCF"
  # checkpointDir = os.path.dirname(checkpointPath)
  isTraining = False

  playDf = getData(1000000)

  numUser = playDf.user_id.max() + 1
  numItem = playDf.artist_id.max() + 1
  num_factor = 32
  epochs = 10
  batchSize = 1024 * 16

  model = buildNeuralCollaborativeFilteringModel(numUser, numItem, num_factor)

  train, test = train_test_split(playDf, test_size=0.2)
  train, val = train_test_split(train, test_size=0.2)
  print(len(train), 'train examples')
  print(len(val), 'validation examples')
  print(len(test), 'test examples')

  testDataset = bootstrapDataset(test, epochs, batchSize, False)
  if isTraining:
    # Create a callback that saves the model's weights
    checkpointCallBack = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointPath,
                                                            save_weights_only=True,
                                                            verbose=1)

    trainDataset = bootstrapDataset(train, epochs, batchSize)
    valDataset = bootstrapDataset(val, epochs, batchSize, False)
    model.fit(trainDataset, validation_data=valDataset, epochs=epochs, callbacks=[checkpointCallBack])

    print("Evaluating trained model...")
    loss, accuracy = model.evaluate(testDataset)
    print("Accuracy", accuracy)
    print("Predict a sample:")

  else:
    model.load_weights(checkpointPath)

    print("Predict a sample:")
    predictions = model.predict(testDataset) #TODO give prediction to one entity only?

    print("predictions shape:", predictions.shape)


if __name__ == "__main__":
  main()
