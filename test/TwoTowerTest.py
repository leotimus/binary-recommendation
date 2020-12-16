import os, sys
import pandas as pd
import tensorflow as tf
import smbclient as smbc

from src.models.TwoTower import TwoTowerModel


def getAAUfilename(filename):
  REPO_PATH = r"\\cs.aau.dk\fileshares\IT703e20"
  return REPO_PATH + "\\" + filename


def splitTrainTest(data, ratio):
  dataSize = len(data)
  trainSetSize = int(dataSize * ratio)
  testSetSize = dataSize - trainSetSize
  shuffled = data.shuffle(dataSize, reshuffle_each_iteration=False)

  train = shuffled.take(trainSetSize)
  test = shuffled.skip(trainSetSize).take(testSetSize)
  return (train, test)


def getOptimizer(optimizerName="Adam", learningRate=0.001):
  optimizerClasses = {
    "Adagrad": tf.optimizers.Adagrad,
    "Adam": tf.optimizers.Adam,
    "Ftrl": tf.optimizers.Ftrl,
    "RMSProp": tf.optimizers.RMSprop,
    "SGD": tf.optimizers.SGD,
  }
  optimizer = optimizerClasses[optimizerName](learning_rate=learningRate)
  return optimizer


def gfData(filename, username, psw, rdZero=False):
  if rdZero:
    cols = ["CUSTOMER_ID", "NORMALIZED_CUSTOMER_ID", "MATERIAL", "PRODUCT_ID", "RATING_TYPE"]
  else:
    cols = ['CUSTOMER_ID', 'MATERIAL']

  with smbc.open_file(getAAUfilename(filename), mode="r", username=username, password=psw) as f:
    data = pd.read_csv(f, names=cols, dtype={"MATERIAL": str, "CUSTOMER_ID": str})
  data.drop(data.index[:1], inplace=True)

  if rdZero:
    data.drop(columns=["NORMALIZED_CUSTOMER_ID", "PRODUCT_ID"], inplace=True)
    data["RATING_TYPE"] = data["RATING_TYPE"].apply(lambda x: float(x))

  # ratings["MATERIAL"] = ratings["MATERIAL"].apply(lambda x : str(x))
  # ratings["CUSTOMER_ID"] = ratings["CUSTOMER_ID"].apply(lambda x : str(x))

  materialId = pd.unique(data["MATERIAL"])
  nbrMaterial = len(materialId)
  usersId = pd.unique(data["CUSTOMER_ID"])
  nbrUser = len(usersId)

  return {"ratings": data, "nbrUser": nbrUser, "nbrMaterial": nbrMaterial, "materialsId": materialId,
          "usersId": usersId}


def topKMetrics(predictions, positives, usersId, itemsId):
  nbrUser = len(usersId)
  nbrItem = len(itemsId)
  total = nbrUser * nbrItem

  real = set(positives)

  tp = 0
  fp = 0
  hits = 0
  for u, topk in predictions:
    hit = False
    for r, i in topk:
      if (u, i) in real:
        tp += 1
        hit = True
      else:
        fp += 1
    if (hit):
      hits += 1
  #			print("hit by user: ",u)
  fn = len(real) - tp
  tn = total - tp - fp - fn
  hitRate = hits / nbrUser
  return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": tp / (tp + fp), "recall": tp / (tp + fn),
          "hitRate": hitRate}


def crossValidation(filenames, k, learningRate, optimiser, loss, epoch, embNum, batchSize, randomZero=False,
                    rdZeroFilenames=None, testBatchSize=5000, semb=64, bname="../result/100kBenchmark"):
  if not os.path.isdir(bname):
    os.mkdir(bname)
  # Load the files for cross-validation.
  dataSets = []
  username = ""  # FIXME
  psw = ""  # FIXME
  print("Loading files", flush=True)
  for filename in filenames:
    dataSets.append(gfData(filename, username, psw))

  rdZeroDataSets = []
  if randomZero:
    for filename in rdZeroFilenames:
      rdZeroDataSets.append(gfData(filename, username, psw, rdZero=True))

  print("Extracting data", flush=True)
  # getting all unique users id and materials id + extracting datasets
  usersId = []
  matId = []
  datas = []
  for dataSet in dataSets:
    usersId.append(pd.Series(dataSet["usersId"]))
    matId.append(pd.Series(dataSet["materialsId"]))
    datas.append(dataSet["ratings"])
  usersId = pd.unique(pd.concat(usersId))
  matId = pd.unique(pd.concat(matId))
  dataSets = datas

  if randomZero:
    datas = []
    for dataSet in rdZeroDataSets:
      datas.append(dataSet["ratings"])

  """dataSet = movieLensData(1,0,1)
  usersId = dataSet["usersId"]
  matId = dataSet["moviesId"]
  datas = dataSet["ratings"]
  datas = datas.rename(columns = {"rating":"RATING_TYPE", "user_id":"CUSTOMER_ID", "movie_id":"MATERIAL"})

  n =  datas.shape[0]//5 + 1  #chunk row size
  datas = [datas[i:i+n] for i in range(0,datas.shape[0],n)]

  dataSets = []
  for d in datas:
    dataSets.append(d.query("RATING_TYPE == 1.0"))"""

  if randomZero:
    dataSets, testDataSets = datas, dataSets  # make the datasets with randomly added zeros the training datasets

  # cross-validation
  res = []
  fullRes = []
  for i in range(len(dataSets)):
    print("cross validation it: " + str(i + 1) + "/" + str(len(dataSets)))
    # creating test set and training set
    testData = dataSets.pop(0)

    if randomZero:
      testData, fakeTestData = testDataSets.pop(0), testData  # change to make the test set wanted

    testSet = tf.data.Dataset.from_tensor_slices(dict(testData))
    trainSet = tf.data.Dataset.from_tensor_slices(dict(pd.concat(dataSets, ignore_index=True)))

    if randomZero:
      # make the change of test set invisible for the rest of the function
      testDataSets.append(testData)
      testData = fakeTestData

    # preparing trainingSet
    print("Shuffuling training set", flush=True)
    trainSet = trainSet.shuffle(len(trainSet), reshuffle_each_iteration=False)
    trainSetCached = trainSet.batch(batchSize).cache()

    # creating model
    model = TwoTowerModel(embNum, len(matId), len(usersId), "CUSTOMER_ID", "MATERIAL", usersId, matId,
                          eval_batch_size=batchSize, loss=loss, rdZero=randomZero, resKey="RATING_TYPE", semb=semb)
    model.compile(optimizer=getOptimizer(optimiser, learningRate=learningRate),
                  loss=tf.keras.losses.BinaryCrossentropy())

    # training
    print("training", flush=True)
    model.fit(trainSetCached, epochs=epoch)

    # testing
    print("testing", flush=True)

    # topk = topKRatings(k, model, usersId, matId, "two tower")
    model.setCandidates(tf.data.Dataset.from_tensor_slices(matId), k)
    pred = model.predict(usersId, batch_size=testBatchSize)
    topk = []
    counter = 0
    for user in tf.data.Dataset.from_tensor_slices(usersId):
      print("\rFormating topK: " + str(counter + 1) + "/" + str(len(usersId)), end="")
      topk.append(
        (str(user.numpy()), [(pred[0][counter][j], str(pred[1][counter][j])) for j in range(len(pred[0][counter]))]))
      counter += 1
    print("", flush=True)
    # print(topk.numpy())
    # print([(str(i["CUSTOMER_ID"].numpy()), str(i["MATERIAL"].numpy()), str(i["RATING_TYPE"].numpy())) for i in testSet])
    # print(len([(str(i["CUSTOMER_ID"].numpy()), str(i["MATERIAL"].numpy()), str(i["RATING_TYPE"].numpy())) for i in testSet]))
    res.append(
      topKMetrics(topk, [(str(i["CUSTOMER_ID"].numpy()), str(i["MATERIAL"].numpy())) for i in testSet], usersId, matId))

    print("Metrics:", res[-1], flush=True)

    # making ready for next it
    dataSets.append(testData)

    if randomZero:
      fullRes.append(topKMetrics(topk, [(str(i["CUSTOMER_ID"].numpy()), str(i["MATERIAL"].numpy())) for i in
                                        tf.data.Dataset.from_tensor_slices(
                                          dict(pd.concat(testDataSets, ignore_index=True)))], usersId, matId))
    else:
      fullRes.append(topKMetrics(topk, [(str(i["CUSTOMER_ID"].numpy()), str(i["MATERIAL"].numpy())) for i in
                                        tf.data.Dataset.from_tensor_slices(
                                          dict(pd.concat(dataSets, ignore_index=True)))], usersId, matId))

    print("Full dataset metrics:", fullRes[-1], flush=True)

  # computing average results
  averageMetrics = {}
  for metrics in res[0]:
    averageMetrics[metrics] = 0
    for itRes in res:
      averageMetrics[metrics] += itRes[metrics]

    averageMetrics[metrics] /= len(dataSets)

  for metrics in fullRes[0]:
    key = "full_" + metrics
    averageMetrics[key] = 0
    for itRes in fullRes:
      averageMetrics[key] += itRes[metrics]

    averageMetrics[key] /= len(dataSets)

  return averageMetrics


if __name__ == "__main__":
  print(sys.argv)
  learningRate = 0.1
  optimiser = "Adagrad"
  splitRatio = 0.8
  loss = None
  # filename = [r"(NEW)CleanDatasets\TT\2m(OG)\ds2_OG(2m)_timeDistributed_1.csv", r"(NEW)CleanDatasets\TT\2m(OG)\ds2_OG(2m)_timeDistributed_2.csv", r"(NEW)CleanDatasets\TT\2m(OG)\ds2_OG(2m)_timeDistributed_3.csv", r"(NEW)CleanDatasets\TT\2m(OG)\ds2_OG(2m)_timeDistributed_4.csv", r"(NEW)CleanDatasets\TT\2m(OG)\ds2_OG(2m)_timeDistributed_5.csv"]
  # rdZeroFilename = [r"(NEW)CleanDatasets\NCF\2m(OG)\ds2_OG(2m)_timeDistributed_1.csv", r"(NEW)CleanDatasets\NCF\2m(OG)\ds2_OG(2m)_timeDistributed_2.csv", r"(NEW)CleanDatasets\NCF\2m(OG)\ds2_OG(2m)_timeDistributed_3.csv", r"(NEW)CleanDatasets\NCF\2m(OG)\ds2_OG(2m)_timeDistributed_4.csv", r"(NEW)CleanDatasets\NCF\2m(OG)\ds2_OG(2m)_timeDistributed_5.csv"]
  # filename = [r"(NEW)CleanDatasets\TT\1m\ds2_1m_timeDistributed_1.csv", r"(NEW)CleanDatasets\TT\1m\ds2_1m_timeDistributed_2.csv", r"(NEW)CleanDatasets\TT\1m\ds2_1m_timeDistributed_3.csv", r"(NEW)CleanDatasets\TT\1m\ds2_1m_timeDistributed_4.csv", r"(NEW)CleanDatasets\TT\1m\ds2_1m_timeDistributed_5.csv"]
  # rdZeroFilename = [r"(NEW)CleanDatasets\NCF\1m\ds2_1m_timeDistributed_1.csv", r"(NEW)CleanDatasets\NCF\1m\ds2_1m_timeDistributed_2.csv", r"(NEW)CleanDatasets\NCF\1m\ds2_1m_timeDistributed_3.csv", r"(NEW)CleanDatasets\NCF\1m\ds2_1m_timeDistributed_4.csv", r"(NEW)CleanDatasets\NCF\1m\ds2_1m_timeDistributed_5.csv"]
  filename = [r"(NEW)CleanDatasets\TT\500k\ds2_500k_timeDistributed_1.csv",
              r"(NEW)CleanDatasets\TT\500k\ds2_500k_timeDistributed_2.csv",
              r"(NEW)CleanDatasets\TT\500k\ds2_500k_timeDistributed_3.csv",
              r"(NEW)CleanDatasets\TT\500k\ds2_500k_timeDistributed_4.csv",
              r"(NEW)CleanDatasets\TT\500k\ds2_500k_timeDistributed_5.csv"]
  rdZeroFilename = [r"(NEW)CleanDatasets\NCF\500k\ds2_500k_timeDistributed_1.csv",
                    r"(NEW)CleanDatasets\NCF\500k\ds2_500k_timeDistributed_2.csv",
                    r"(NEW)CleanDatasets\NCF\500k\ds2_500k_timeDistributed_3.csv",
                    r"(NEW)CleanDatasets\NCF\500k\ds2_500k_timeDistributed_4.csv",
                    r"(NEW)CleanDatasets\NCF\500k\ds2_500k_timeDistributed_5.csv"]
  # filename = [r"(NEW)CleanDatasets\TT\10m\ds2_10m_timeDistributed_1.csv", r"(NEW)CleanDatasets\TT\10m\ds2_10m_timeDistributed_2.csv", r"(NEW)CleanDatasets\TT\10m\ds2_10m_timeDistributed_3.csv", r"(NEW)CleanDatasets\TT\10m\ds2_10m_timeDistributed_4.csv", r"(NEW)CleanDatasets\TT\10m\ds2_10m_timeDistributed_5.csv"]
  epoch = 3
  semb = 50
  embNum = 75
  batchSize = 1000
  testBatchSize = 5000
  k = 10
  randomZero = False
  bname = "../result/benchmarking-project-ma1/100kBenchmark"
  for i in range(len(sys.argv)):
    if sys.argv[i] == "data":
      filename = sys.argv[i + 1]
    elif sys.argv[i] == "loss":
      loss = sys.argv[i + 1]
    elif sys.argv[i] == "epoch":
      epoch = int(sys.argv[i + 1])
    elif sys.argv[i] == "lrate":
      learningRate = float(sys.argv[i + 1])
    elif sys.argv[i] == "ratio":
      splitRatio = float(sys.argv[i + 1])
    elif sys.argv[i] == "k":
      k = float(sys.argv[i + 1])
    elif sys.argv[i] == "opti":
      optimiser = sys.argv[i + 1]
    elif sys.argv[i] == "randomZero":
      randomZero = True
    elif sys.argv[i] == "bname":
      bname = "../result/benchmarking-project-ma1/" + sys.argv[i + 1]

  res = crossValidation(
    filename,
    k,
    learningRate,
    optimiser,
    loss,
    epoch,
    embNum,
    batchSize,
    rdZeroFilenames=rdZeroFilename,
    randomZero=randomZero,
    testBatchSize=testBatchSize,
    semb=semb,
    bname=bname
  )
  print("Average metrics:", res, flush=True)
  with open("../result/twoTowerResult_deep", "a") as f:
    f.write(
      "k: " + str(k) + ", learning rate: " + str(learningRate) + ", optimiser: " + optimiser + ", splitRatio: " + str(
        splitRatio) + ", loss: " + str(loss) + ", filename: " + str(filename) + ", use randomly added zeros: " + str(
        randomZero) + ", randomly added zeros filename: " + str(rdZeroFilename) + ", epoch: " + str(
        epoch) + ", nbr embedings: " + str(embNum) + ", second emb: " + str(semb) + ", batchSize: " + str(
        batchSize) + "\n")
    f.write(str(res) + "\n")
  # with open("../result/twoTowerResult", "a") as f:
  # f.write("k: " + str(k) + ", learning rate: " + str(learningRate) + ", optimiser: " + optimiser + ", splitRatio: " + str(splitRatio) + ", loss: " + str(loss) + ", filename: " + str(filename) + ", epoch: " + str(epoch) + "nbr embedings: " + str(embNum) + ", batchSize: " + str(batchSize) + "\n")
  # f.write(str(res) + "\n")
  print("Done", flush=True)

  """#data = movieLensData(1,0,0)
  data = gfData(filename)
  #print(dict(data["ratings"]))
  ratings = tf.data.Dataset.from_tensor_slices(dict(data["ratings"]))
  trainSet, testSet = splitTrainTest(ratings, splitRatio)
  #model = TwoTowerModel(embNum, data["nbrMovie"], data["nbrUser"], "user_id", "movie_id", "rating", data["usersId"], data["moviesId"], ratings)
  model = TwoTowerModel(embNum, data["nbrMaterial"], data["nbrUser"], "CUSTOMER_ID", "MATERIAL", "is_real", data["usersId"], data["materialsId"], ratings, eval_batch_size = batchSize, loss = loss)
  threshold = 0.5
  #model.compile(optimizer = getOptimizer("Adam",learningRate = 0.01), metrics=["MAE","MSE",tf.keras.metrics.BinaryAccuracy(threshold = threshold), tf.keras.metrics.TrueNegatives(threshold), tf.keras.metrics.TruePositives(threshold), tf.keras.metrics.FalseNegatives(threshold), tf.keras.metrics.FalsePositives(threshold)])
  model.compile(optimizer = getOptimizer(optimiser, learningRate = learningRate))

  trainSetCached = trainSet.batch(batchSize).cache()
  #trainSetCached = trainSet.batch(80000)
  #tf.keras.utils.plot_model(model, expand_nested = True)
  model.fit(trainSetCached, epochs = epoch)
  print("test")
  #res = model.evaluate(testSet.batch(batchSize).cache(), return_dict=True)
  #with open("../result/twoTowerResult", "a") as f:
  #	f.write("learning rate: " + str(learningRate) + ", optimiser: " + optimiser + ", splitRatio: " + str(splitRatio) + ", loss: " + str(loss) + ", filename: " + filename + ", epoch: " + str(epoch) + "nbr embedings: " + str(embNum) + ", batchSize: " + str(batchSize))
  #	f.write(str(res))
  #model.evaluate(testSet.batch(40000), return_dict=True)
  #raise Exception
  topk = topKRatings(10, model, data["usersId"], data["materialsId"], "two tower")
  print(topKMetrics(topk, [(str(int(i["CUSTOMER_ID"].numpy())), str(int(i["MATERIAL"].numpy()))) for i in testSet], data["usersId"], data["moviesId"]))"""
