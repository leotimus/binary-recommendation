
import numpy as np
import pandas as pd
from os import path
from collections import OrderedDict
from tqdm import tqdm
from typing import Dict

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Embedding, Flatten, Input, Lambda
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

tf.__version__

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DIR_DATA = 'data'
DIR_MODEL = 'models'

"""### Get data"""

df_full = pd.read_csv(path.join(DIR_DATA, 'training_ratings_for_kaggle_comp.csv'))
df_full.head(1)

"""### Build the references

I'm planning to use the `Embedding` layer, so I need to link real ids of the users and movies to the order ones.
"""

sorted(df_full.movie.unique())[-10:]

sorted(df_full.user.unique())[:10]

unique_users = df_full.user.unique()
user_ids = dict(zip(unique_users, np.arange(unique_users.shape[0], dtype=np.int32)))

unique_movies = df_full.movie.unique()
movie_ids = dict(zip(unique_movies, np.arange(unique_movies.shape[0], dtype=np.int32)))

df_full['user_id'] = df_full.user.apply(lambda u: user_ids[u])
df_full['movie_id'] = df_full.movie.apply(lambda m: movie_ids[m])

df_full.head(1)

"""### Train/test split

Here the main idea is to extract some movies for users who have a big amount of positive reviews into the test subtest. 
I extract 2 movies for each user who have more than 20 positive reviews. This test subset won't be used during training, 
but these movies should appear in the top recommendations for each user accordingly.

#### Test subset
"""

tmp_test = df_full[df_full.rating > 4]
tmp_test = tmp_test.groupby('user').movie.count().reset_index()

conditions = (df_full.user.isin(tmp_test[tmp_test.movie > 20].user)) & (df_full.rating > 4)
df_test = df_full[conditions].groupby('user').head(2).reset_index()

del df_test['index']

ground_truth_test = df_test.groupby('user_id').movie_id.agg(list).reset_index()

ground_truth_test.head(1)

"""#### Training subset"""

df_train = pd.concat([df_full, df_test]).drop_duplicates(keep=False)

# The assumption is that the recommendations should as many as possible high ranked movies
# that a specific user has already watched.

ground_truth_train = df_train[df_train.rating > 3].groupby('user_id').movie_id.agg(list).reset_index()

ground_truth_train.head(1)

"""### Building triplets

Bayers Personalized Ranking requires for the training a triplet of the user, positive item and negative item. For each user, I create a pair of each positive ranked movie (the rank is higher than 3) with all negative movies (the rank is equal  3 and lower than).
"""

df_triplest = pd.DataFrame(columns=['user_id', 'positive_m_id', 'negative_m_id'])

# Commented out IPython magic to ensure Python compatibility.
data = []
users_without_data = []

for user_id in tqdm(df_train.user_id.unique()):
    positive_movies = df_train[(df_train.user_id == user_id) & (df_train.rating > 3)].movie_id.values
    negative_movies = df_train[(df_train.user_id == user_id) & (df_train.rating <= 3)].movie_id.values

    if negative_movies.shape[0] == 0 or positive_movies.shape[0] == 0:
        users_without_data.append(user_id)
        continue


    for positive_movie in positive_movies:
        for negative_movie in negative_movies:
            data.append({'user_id': user_id, 'positive_m_id': positive_movie, 'negative_m_id': negative_movie})

df_triplest = df_triplest.append(data, ignore_index=True)
df_triplest.to_csv(path.join(DIR_DATA, 'triplets.csv'), index=False)

"""### BPR NN"""

num_users = unique_users.shape[0]
num_items = unique_movies.shape[0]

unique_movie_ids = list(df_full.movie_id.unique())

"""### Build a model"""


def bpr_predict(model: Model, user_id: int, item_ids: list, user_layer='user_embedding', item_layer='item_embedding'):
  """
  Predict by multiplication user vector by item matrix

  :return: list of the scores
  """
  user_vector = model.get_layer(user_layer).get_weights()[0][user_id]
  item_matrix = model.get_layer(item_layer).get_weights()[0][item_ids]

  scores = (np.dot(user_vector, item_matrix.T))

  return scores


@tf.function
def identity_loss(_, y_pred):
  return tf.math.reduce_mean(y_pred)


@tf.function
def bpr_triplet_loss(X: dict):
  """
  Calculate triplet loss - as higher the difference between positive interactions
  and negative interactions as better

  :param X: X contains the user input, positive item input, negative item input
  :return:
  """
  positive_item_latent, negative_item_latent, user_latent = X

  positive_interactions = tf.math.reduce_sum(tf.math.multiply(user_latent, positive_item_latent), axis=-1,
                                             keepdims=True)
  negative_interactions = tf.math.reduce_sum(tf.math.multiply(user_latent, negative_item_latent), axis=-1,
                                             keepdims=True)

  return tf.math.subtract(tf.constant(1.0), tf.sigmoid(tf.math.subtract(positive_interactions, negative_interactions)))


def out_shape(shapes):
  return shapes[0]


def build_model(num_users: int, num_items: int, latent_dim: int) -> Model:
  """
  Build a model for Bayesian personalized ranking

  :param num_users: a number of the unique users
  :param num_items: a number of the unique movies
  :param latent_dim: vector length for the latent representation
  :return: Model
  """
  user_input = Input((1,), name='user_input')

  positive_item_input = Input((1,), name='positive_item_input')
  negative_item_input = Input((1,), name='negative_item_input')
  # One embedding layer is shared between positive and negative items
  item_embedding_layer = Embedding(num_items, latent_dim, name='item_embedding', input_length=1)

  positive_item_embedding = Flatten()(item_embedding_layer(positive_item_input))
  negative_item_embedding = Flatten()(item_embedding_layer(negative_item_input))

  user_embedding = Embedding(num_users, latent_dim, name='user_embedding', input_length=1)(user_input)
  user_embedding = Flatten()(user_embedding)

  triplet_loss = Lambda(bpr_triplet_loss, output_shape=out_shape)([positive_item_embedding,
                                                                   negative_item_embedding,
                                                                   user_embedding])

  model = Model(inputs=[positive_item_input, negative_item_input, user_input], outputs=triplet_loss)

  return model


latent_dim = 350
batch_size = 256
num_epochs = 1
lr = 0.001

model = build_model(num_users, num_items, latent_dim)
model.compile(loss=identity_loss, optimizer=Adam(learning_rate=lr))

trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])

print('Total number of parameters: {:,}'.format(trainable_count + non_trainable_count))
print('Trainable number of parameters: {:,}'.format(trainable_count))
print('Non-trainable number of parameters: {:,}'.format(non_trainable_count))

print('Training data length: {:,}'.format(df_triplest.shape[0]))

# Commented out IPython magic to ensure Python compatibility.
#
X = {
  'user_input': tf.convert_to_tensor(np.asarray(df_triplest.user_id).astype(np.float32)),
  'positive_item_input': tf.convert_to_tensor(np.asarray(df_triplest.positive_m_id).astype(np.float32)),
  'negative_item_input': tf.convert_to_tensor(np.asarray(df_triplest.negative_m_id).astype(np.float32))
}

model.fit(X,
          tf.ones(df_triplest.shape[0]),
          batch_size=batch_size,
          epochs=num_epochs)

model.save(path.join(DIR_DATA, 'model.h5'))

"""### Evaluation"""


def full_auc(model: Model, ground_truth: Dict[int, list], items: list) -> float:
  """
  Measure AUC for model and ground truth for all items

  :param model:
  :param ground_truth: dictionary of the users and the high ranked movies for the specific user
  :param items: a list of the all available movies
  :return: AUC
  """

  number_of_items = len(items)
  scores = []

  for user_id, true_item_ids in ground_truth:
    predictions = bpr_predict(model, user_id, items)
    grnd = np.zeros(number_of_items, dtype=np.int32)

    for p in true_item_ids:
      index = items.index(p)
      grnd[index] = 1

    if true_item_ids:
      scores.append(roc_auc_score(grnd, predictions))

  return sum(scores) / len(scores)


def mean_average_precision_k(model: Model,
                             ground_truth: Dict[int, list],
                             items: list,
                             k=100) -> float:
  """
  Calculate mean eavarage precission per user

  :param model:
  :param ground_truth: dictionary of the users and the high ranked movies for the specific user
  :param items: a list of the all available movies
  :param k: top N recommendations per user
  :return: mean eavarage precission
  """
  scores = []

  for user, actual in ground_truth:
    predictions = bpr_predict(model, user, items)
    predictions = dict(zip(items, predictions))
    predictions = sorted(predictions.items(), key=lambda kv: kv[1], reverse=True)[:k]
    predictions = list(OrderedDict(predictions).keys())

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predictions):
      if p in actual:
        num_hits += 1.0
        score += num_hits / (i + 1.0)

    score = score / min(len(actual), k)
    scores.append(score)

  return np.mean(scores)


"""#### Train"""

print(f'AUC train: {full_auc(model, ground_truth_train.values, unique_movie_ids)}')

print(f'Mean average precision train: {mean_average_precision_k(model, ground_truth_train.values, unique_movie_ids)}')

"""#### Test"""

print(f'AUC test: {full_auc(model, ground_truth_test.values, unique_movie_ids)}')

print(f'Mean average precision test: {mean_average_precision_k(model, ground_truth_test.values, unique_movie_ids)}')

