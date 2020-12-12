from src.models.BPRModel import BPRModel

if __name__ == '__main__':
  model = BPRModel()
  model.train('data/sdata.csv', rowLimit=1000, metricDict={}, distributedConfig=None)
