import numpy as np
import pandas as pd
import keras
import keras.utils
import tensorflow as tf
import time
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

start_time = time.time()

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

print ('Loading dataset...')

dataset = pd.read_csv('//cs.aau.dk/Fileshares/IT703e20/CleanDatasets/binary_cleaned_incl_customers.csv', header=0, names=['index', 'customer_id', 'material_id', 'is_real'])
#dataset = pd.read_csv('D:/ML/dataset/binary_cleaned_incl_customers.csv', header=0, names=['index', 'customer_id', 'material_id', 'is_real'])

print ('Dataset loaded')

num_users = len(dataset.customer_id.unique())
num_movies = len(dataset.material_id.unique())

#shuffle here
dataset = shuffle(dataset)

dataset.drop('index', axis=1, inplace=True)
train, test = train_test_split(dataset, test_size=0.2)

#Build the model
print ("Building model")
latent_dim = 10

movie_input = Input(shape=[1],name='movie-input')
user_input = Input(shape=[1], name='user-input')

# MLP Embeddings
movie_embedding_mlp = Embedding(num_movies + 1, latent_dim, name='movie-embedding-mlp')(movie_input)
movie_vec_mlp = Flatten(name='flatten-movie-mlp')(movie_embedding_mlp)

user_embedding_mlp = Embedding(num_users + 1, latent_dim, name='user-embedding-mlp')(user_input)
user_vec_mlp = Flatten(name='flatten-user-mlp')(user_embedding_mlp)

# MF Embeddings
movie_embedding_mf = Embedding(num_movies + 1, latent_dim, name='movie-embedding-mf')(movie_input)
movie_vec_mf = Flatten(name='flatten-movie-mf')(movie_embedding_mf)

user_embedding_mf = Embedding(num_users + 1, latent_dim, name='user-embedding-mf')(user_input)
user_vec_mf = Flatten(name='flatten-user-mf')(user_embedding_mf)

# MLP layers
concat = tf.keras.layers.Concatenate(axis=-1)([movie_vec_mlp, user_vec_mlp])
concat_dropout = Dropout(0.2)(concat)
fc_1 = Dense(100, name='fc-1', activation='sigmoid')(concat_dropout)
fc_1_bn = BatchNormalization(name='batch-norm-1')(fc_1)
fc_1_dropout = Dropout(0.2)(fc_1_bn)
fc_2 = Dense(50, name='fc-2', activation='sigmoid')(fc_1_dropout)
fc_2_bn = BatchNormalization(name='batch-norm-2')(fc_2)
fc_2_dropout = Dropout(0.2)(fc_2_bn)

# Prediction from both layers
pred_mlp = Dense(10, name='pred-mlp', activation='sigmoid')(fc_2_dropout)
pred_mf = tf.keras.layers.Dot(axes=1)([movie_vec_mf, user_vec_mf])
combine_mlp_mf = tf.keras.layers.Concatenate(axis=-1)([pred_mf, pred_mlp])

# Final prediction
result = Dense(1, name='result', activation='sigmoid')(combine_mlp_mf)

model = Model([user_input, movie_input], result)
model.compile('adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryCrossentropy(), 'mse', 'mae', tf.keras.metrics.FalseNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.TruePositives(), tf.keras.metrics.Accuracy(), tf.keras.metrics.BinaryAccuracy()])

#train and evaluate the model
print ('Training model')
history = model.fit([train.customer_id, train.material_id], train.is_real, epochs=10, batch_size=50000, shuffle=True)
pd.Series(history.history['loss']).plot(logy=True)
plt.xlabel("Epoch")
plt.ylabel("Train Error")
plt.show()
print ('Training model...Done')

print ('Predict model')
y_hat = np.round(model.predict([test.customer_id, test.material_id]), decimals=2)
y_true = test.is_real
mean_absolute_error(y_true, y_hat)
print ('Predict model...Done')

print ('Evaluate model')
model.evaluate([test.customer_id, test.material_id], test.is_real)
print ('Evaluate model...Done')