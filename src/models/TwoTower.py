import tensorflow as tf
import tensorflow_recommenders as tfrs


class TwoTowerModel(tf.keras.Model):
  def __init__(self, embedDim, nbrItem, nbrUser, userKey, itemKey, usersId, itemsId, eval_batch_size=8000, loss=None,
               rdZero=False, resKey=None, semb=100):
    super().__init__(self)
    self.embedDim = embedDim
    self.nbrItem = nbrItem
    self.nbrUser = nbrUser
    self.userKey = userKey
    self.itemKey = itemKey
    self.resKey = resKey
    self.bruteForceLayer = tfrs.layers.factorized_top_k.Streaming()
    self.eval_batch_size = eval_batch_size
    self.rdZero = rdZero

    self.userTowerIn = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=usersId)
    self.userTowerOut = tf.keras.layers.Embedding(nbrUser + 2, embedDim)
    self.itemTowerIn = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=itemsId)
    self.itemTowerOut = tf.keras.layers.Embedding(nbrItem + 2, embedDim)

    self.userTower = tf.keras.Sequential([self.userTowerIn, self.userTowerOut, tf.keras.layers.Dense(semb)])
    self.itemTower = tf.keras.Sequential([self.itemTowerIn, self.itemTowerOut, tf.keras.layers.Dense(semb)])

    if rdZero:
      self.outputLayer = tf.keras.layers.Dot(axes=-1)
      self.computeLoss = self.computeLossRdZero
    else:
      self.task = tfrs.tasks.Retrieval(loss=loss)
      self.computeLoss = self.computeLossTfrs

  def call(self, info):
    usersCaracteristics = self.userTower(info)
    return self.bruteForceLayer(usersCaracteristics)

  def setCandidates(self, items, k):
    self.bruteForceLayer = tfrs.layers.factorized_top_k.BruteForce(k=k)
    self.bruteForceLayer.index(
      candidates=items.batch(self.eval_batch_size).map(self.itemTower),
      identifiers=items
    )

  """def getTopK(self, users, k):
    self.bruteForceLayer(
          query_embeddings = self.userTower(users), 
          k = k
        )"""

  def computeEmb(self, info):
    usersCaracteristics = self.userTower(info[self.userKey])
    itemCaracteristics = self.itemTower(info[self.itemKey])
    return (usersCaracteristics, itemCaracteristics)

  def computeLossTfrs(self, usersCaracteristics, itemCaracteristics, info):
    return self.task(usersCaracteristics, itemCaracteristics, compute_metrics=False, training=True,
                     candidate_ids=info[self.itemKey])

  def computeLossRdZero(self, usersCaracteristics, itemCaracteristics, info):
    pred = tf.keras.activations.sigmoid(self.outputLayer([usersCaracteristics, itemCaracteristics]))
    return self.compiled_loss(info[self.resKey], pred)

  def train_step(self, info):
    with tf.GradientTape() as tape:
      # pred = self(info, training = True)
      # loss = self.compiled_loss(info[self.resKey], pred)
      usersCaracteristics, itemCaracteristics = self.computeEmb(info)
      loss = self.computeLoss(usersCaracteristics, itemCaracteristics, info)

    # print(self.trainable_variables)
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    # self.compiled_metrics.update_state(info[self.resKey], pred)
    metrics = {m.name: m.result() for m in self.metrics}
    metrics["loss"] = loss
    return metrics

  def test_step(self, info):
    # pred = self(info)
    usersCaracteristics, itemCaracteristics = self.computeEmb(info)
    loss = self.task(usersCaracteristics, itemCaracteristics, candidate_ids=info[self.itemKey])
    # self.compiled_metrics.update_state(info[self.resKey], pred)
    metrics = {m.name: m.result() for m in self.metrics}
    metrics["loss"] = loss
    return metrics
