import numpy as np
import pandas as pd
import keras
import keras.utils
import tensorflow as tf
import time
import os
import neptune
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from src.AAUFile import *
from getpass import getpass
import smbclient as smbc
import trainers.topKmetrics as trainerTop
#from trainers.topKmetrics import topKMetrics


start_time = time.time()

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

print ('Loading dataset...')

# path = "/home/user/data/"
# all_files = glob.glob(os.path.join(path, "data_*.csv"))
# df_merged = (pd.read_csv(f, sep=',') for f in all_files)
# df_merged   = pd.concat(df_from_each_file, ignore_index=True)
# df_merged.to_csv( "merged.csv")
#  def crossValidation(filenames, k, learningRate, optimiser, loss, epoch, embNum, batchSize):
#   	#Load the files for cross-validation.
#  	dataSets = []
#  	username = input("username:")
#  	psw = getpass()
#  	print("Loading files")
#  	for filename in filenames:
#  		dataSets.append(gfData(filename, username, psw))
	
#  	print("Loading done.")
	
# 	#getting all unique users id and materials id
# 	usersId = []
# 	matId = []
# 	datas = []
# 	for dataSet in dataSets:
# 		usersId.append(pd.Series(dataSet["usersId"]))
# 		matId.append(pd.Series(dataSet["materialsId"]))
# 		datas.append(dataSet["ratings"])
# 	usersId = pd.unique(pd.concat(usersId))
# 	matId = pd.unique(pd.concat(matId))
# 	dataSets = datas
	
# 	#cross-validation
# 	res = []
# 	for i in range(len(dataSets)):
# 		print("cross validation it: " + str(i) + "/" + str(len(dataSets)))
# 		#creating test set and training set
# 		testData = dataSets.pop(0)
# 		testSet = tf.data.Dataset.from_tensor_slices(dict(testData))
# 		trainSet = tf.data.Dataset.from_tensor_slices(dict(pd.concat(dataSets, ignore_index=True)))

# with smbc.open_file((r""), mode="r", username=input("username: "), password=getpass()) as f:
#     train = pd.read_csv(f, header=0, names=['customer_id', 'normalized_customer_id', 'material', 'product_id', 'rating_type'])

# with smbc.open_file((r""), mode="r", username=input("username: "), password=getpass()) as f:
#     test = pd.read_csv(f, header=0, names=['customer_id', 'normalized_customer_id', 'material', 'product_id', 'rating_type'])

#dataset = pd.read_csv('D:/ML/dataset/100kNew.csv', header=0, names=['customer_id', 'normalized_customer_id', 'material', 'product_id', 'rating_type'])
train = pd.read_csv('', header=0, names=['customer_id', 'normalized_customer_id', 'material', 'product_id', 'rating_type'])
test = pd.read_csv('', header=0, names=['customer_id', 'normalized_customer_id', 'material', 'product_id', 'rating_type'])

print ('Dataset loaded')
#frames = (train, test)
mergeddata_datasets = train.append(test)

num_customers = len(mergeddata_datasets.normalized_customer_id.unique())
#num_materials = len(dataset.product_id.unique())
num_materials = mergeddata_datasets.product_id.max()

#shuffle here
train = shuffle(train)

#dataset.drop('index', axis=1, inplace=True)
train.drop(columns=['customer_id','material'])
test.drop(columns=['customer_id','material'])
#train, test = train_test_split(dataset, test_size=0.2)

#print (train.Head())

neptune.init(
    api_token="=",
    project_qualified_name=""
)

neptune.set_project('')
neptune.create_experiment(name='great-idea')


class NeptuneMonitor(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for metric_name, metric_value in logs.items():
            neptune.log_metric(metric_name, metric_value)

#Build the model
print ("Building model")
latent_dim = 10

material_input = Input(shape=[1],name='material-input')
customer_input = Input(shape=[1], name='customer-input')

# MLP Embeddings
material_embedding_mlp = Embedding(num_materials + 1, latent_dim, name='material-embedding-mlp')(material_input)
material_vec_mlp = Flatten(name='flatten-material-mlp')(material_embedding_mlp)

customer_embedding_mlp = Embedding(num_customers + 1, latent_dim, name='customer-embedding-mlp')(customer_input)
customer_vec_mlp = Flatten(name='flatten-customer-mlp')(customer_embedding_mlp)

# MF Embeddings
material_embedding_mf = Embedding(num_materials + 1, latent_dim, name='material-embedding-mf')(material_input)
material_vec_mf = Flatten(name='flatten-material-mf')(material_embedding_mf)

customer_embedding_mf = Embedding(num_customers + 1, latent_dim, name='customer-embedding-mf')(customer_input)
customer_vec_mf = Flatten(name='flatten-customer-mf')(customer_embedding_mf)

PARAMS = {'epoch_nr': 20,
          'batch_size': 50000,
          'lr': 0.005,
          'momentum': 0.4,
          'unit_nr1': 100,
          'unit_nr1': 50,
          'dropout': 0.2}

# MLP layers
concat = tf.keras.layers.Concatenate(axis=-1)([material_vec_mlp, customer_vec_mlp])
concat_dropout = Dropout(0.2)(concat)
fc_1 = Dense(100, name='fc-1', activation='sigmoid')(concat_dropout)
fc_1_bn = BatchNormalization(name='batch-norm-1')(fc_1)
fc_1_dropout = Dropout(PARAMS['dropout'])(fc_1_bn)
fc_2 = Dense(50, name='fc-2', activation='sigmoid')(fc_1_dropout)
fc_2_bn = BatchNormalization(name='batch-norm-2')(fc_2)
fc_2_dropout = Dropout(PARAMS['dropout'])(fc_2_bn)

# Prediction from both layers
pred_mlp = Dense(10, name='pred-mlp', activation='sigmoid')(fc_2_dropout)
pred_mf = tf.keras.layers.Dot(axes=1)([material_vec_mf, customer_vec_mf])
combine_mlp_mf = tf.keras.layers.Concatenate(axis=-1)([pred_mf, pred_mlp])

# Final prediction
result = Dense(1, name='result', activation='sigmoid')(combine_mlp_mf)
optimizer = keras.optimizers.Adam(lr=PARAMS['lr'])
model = Model([customer_input, material_input], result)
model.compile(optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryCrossentropy(), 'mse', 'mae', tf.keras.metrics.FalseNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.TruePositives(), tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.TopKCategoricalAccuracy(k=10)])

#checkpoint_path = "training_1/cp.ckpt"
#checkpoint_dir = os.path.dirname('D:/ML/dataset/savedModels')
#train and evaluate the model
print ('Training model')
#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
 #                                                save_weights_only=True,
    #                                             verbose=1)

history = model.fit([train.normalized_customer_id, train.product_id], train.rating_type, epochs=PARAMS['epoch_nr'], batch_size=PARAMS['batch_size'], shuffle=True, verbose=2, callbacks=[NeptuneMonitor()])
model.save('D:/ML/dataset/savedModels/mymodel')
pd.Series(history.history['loss']).plot(logy=True)
plt.xlabel("Epoch")
plt.ylabel("Train Error")
fig = plt.gcf()
#plt.show()
plt.draw()
fig.savefig('D:/ML/dataset/savedModels/figure.pdf')
print ('Training model...Done')

print ('Predict model')
y_hat = np.round(model.predict([test.normalized_customer_id, test.product_id], verbose=1), decimals=2)
y_true = test.rating_type
mean_absolute_error(y_true, y_hat)
print ('Predict model...Done')

print ('Evaluate model')
model.evaluate([test.normalized_customer_id, test.product_id], test.rating_type, verbose=2, callbacks=[NeptuneMonitor()])
print ('Evaluate model...Done')

# def TopKAccuracy(k):
#     TopK = {}
#     for c in dataset.customer_id.unique():
#       predictions: []
#       for m in dataset.material_id.unique():
#           predictions.append((model.predict((c,m)), (c,m)))
#       predictions.sort()
#       TopK [c] = predictions [:k]

num_customers = len(mergeddata_datasets.normalized_customer_id.unique())
num_materials = mergeddata_datasets.product_id.max()

unique_customers = mergeddata_datasets.normalized_customer_id.unique()
unique_products = mergeddata_datasets.product_id.unique()

res= []

res.append(trainerTop.topKMetrics(10, ([test.normalized_customer_id, test.product_id]), unique_customers, unique_products))
print(res[-1])
#results = topk.topKMetrics(top_10_ratings, test_set, actual_user_ids, actual_item_ids)