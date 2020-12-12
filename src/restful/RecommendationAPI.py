from flask_oauthlib.provider import OAuth2Provider
from flask import Flask
from flask import jsonify
from flask import request

from src.models import RModel
from src.models.NCFModel import NCFModel
from src.models.NeuMFModel import NeuMFModel

app = Flask(__name__)
activeModel:RModel

# TODO oauth2 for strong security
# oauth = OAuth2Provider(app)
app.config["DEBUG"] = True


@app.route('/api/recommendation/<int:userId>/<int:numberOfItem>', methods=['GET'])
def getProductRecommendationForUser(userId, numberOfItem):
  # Call model & feed the recommendation here
  global activeModel
  return jsonify(activeModel.predictForUser(userId, numberOfItem))

@app.route('/api/users', methods=['GET'])
def getUsers():
  # Call model & feed the recommendation here
  global activeModel
  return jsonify(activeModel.getPredictableUsers())


@app.route('/api/models', methods=['GET'])
def getSupportedModels():
  # Call model & feed the recommendation here
  return str('NeuMFModel, NCFModel')

@app.route('/api/models/<operation>/<model>', methods=['POST'])
# operation in [train, active...]
# model in [NeuFM, MF...]
def operateOnModel(operation, model):
  # Call model & feed the recommendation here
  data = request.json

  global activeModel
  if operation == 'active':
    activeModel = getModelByName(model)
    activeModel.restoreFromLatestCheckPoint()
    return {'result': 'ok', 'active model': model}

  elif operation == 'train':
    trainingModel = getModelByName(model)
    if not trainingModel.readyToTrain():
      return {'result': 'error', 'message': 'model not ready to train'}

    return trainingModel.train(data['path'], data['rowLimit'], {})

  return 'Triggered operation {} on {} model without specific return value'.format(operation, model)

def getModelByName(model: str) -> RModel:
  if model == 'NeuMFModel':
    return NeuMFModel()
  if model == 'NCFModel':
    return NCFModel()
  return None

app.run()
