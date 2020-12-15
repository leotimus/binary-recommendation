from src.models.NeuMFModel import NeuMFModel

model = NeuMFModel()
model.batchSize = 1024 * 32
model.train(r'\\cs.aau.dk\Fileshares\IT703e20\(OLD)CleanDatasets\with_0s\MCQ with 0s and INT PRODUCT_ID.csv', 50000)
