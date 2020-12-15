import os
import json

from src.models.NeuMFModel import NeuMFModel

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.pop('TF_CONFIG', None)

os.environ['TF_CONFIG'] = '{"cluster": {"worker": ["localhost:12345", "localhost:23456"]}, "task": {"type": "worker", "index": 1} }'

config = json.loads(os.environ['TF_CONFIG'])

model = NeuMFModel()
# model.train('data/sdata.csv', 50000, {}, config)
model.train('data/sdata.csv', 300000, {}, config)

# =====================
