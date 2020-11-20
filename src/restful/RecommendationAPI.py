from flask_oauthlib.provider import OAuth2Provider
from flask import Flask

from src.models import RModel
from src.models.NeuMFModel import NeuMFModel

app = Flask(__name__)
activeModel:RModel

# TODO oauth2 for strong security
# oauth = OAuth2Provider(app)
app.config["DEBUG"] = True


@app.route('/api/recommendation/<userId>', methods=['GET'])
def getProductRecommendationForUser(userId):
  # Call model & feed the recommendation here
  global activeModel
  return activeModel.predictForUser(userId)

@app.route('/api/users', methods=['GET'])
def getUsers():
  # Call model & feed the recommendation here
  global activeModel
  return activeModel.getPredictableUsers()


@app.route('/api/models', methods=['GET'])
def getSupportedModels():
  # Call model & feed the recommendation here
  return ['MF', 'NeuMF', 'FM', 'NeuFM']

@app.route('/api/models/<operation>/<model>', methods=['POST'])
# operation in [train, active...]
# model in [NeuFM, MF...]
def operateOnModel(operation, model):
  # Call model & feed the recommendation here
  global activeModel
  if operation == 'active':
    activeModel = NeuMFModel('NeuMFModel')

  elif operation == 'train':

    trainingModel = NeuMFModel('NeuMFModel')
    trainingModel.train('data/sdata.csv', 10000, {})

    activeModel = trainingModel #TODO remove

  return 'Triggered operation {} on {} model'.format(operation, model)

app.run()
