import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

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
  df = pd.read_csv('data/lastfm_play.csv', nrows=rows)
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

  return model


def bootstrapDataset(df, negRatio=3., batchSize=128):
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

  ds = (tf.data.Dataset
        .from_tensor_slices((X, Y))
        .batch(batchSize))

  return ds


def main():
  playDf = getData(600000)

  numUser = playDf.user_id.max() + 1
  numItem = playDf.artist_id.max() + 1
  num_factor = 32

  model = buildNeuralCollaborativeFilteringModel(numUser, numItem, num_factor)

  model.compile(Adam(1e-3), loss=BinaryCrossentropy(), metrics=[BinaryAccuracy()])

  #### TRAIN ####
  num_epoch = 10
  batch_size = 1024 * 16
  for i in range(num_epoch):
    print(f"{i + 1}th epoch :")
    dataset = bootstrapDataset(playDf, num_epoch, batch_size)
    model.fit(dataset)

  model.predict()

  #### RUN FROM CHECKPOINT ####




if __name__ == "__main__":
  main()
