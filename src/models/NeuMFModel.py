from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Concatenate, Dense, Embedding, Dropout, BatchNormalization, Dot
from tensorflow.keras.models import Model

from src.models.RModel import RModel

MODEL_TO_DOT_PNG = 'export/model.png'
MODEL_TRAIN_PLOT = 'export/plot.png'
USER_FEATURE = 'CUSTOMER_ID'
PRODUCT_FEATURE = 'PRODUCT_ID'
CP_PATH = 'checkpoints/NeuMF02/cp'


class NeuMFModel(RModel):

  def __init__(self):
    super().__init__('NeuMFModel')

  def buildModel(self, numUser, numItem, numFactor):
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
    output = Dense(1, activation='sigmoid')(combine)

    inputs = [userId, itemId]
    model = Model(inputs, output, name=self.modelName)

    model.compile(Adam(1e-3), loss='mean_squared_error',
                  metrics=RModel.METRICS)

    return model
