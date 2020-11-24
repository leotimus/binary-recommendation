from flask_oauthlib.provider import OAuth2Provider
from flask import Flask
from flask import jsonify
from flask import request

from src.models import RModel
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
  return str('MF, NeuMF, FM, NeuFM')

@app.route('/api/models/<operation>/<model>', methods=['POST'])
# operation in [train, active...]
# model in [NeuFM, MF...]
def operateOnModel(operation, model):
  # Call model & feed the recommendation here
  data = request.json

  global activeModel
  if operation == 'active':

    activeModel = NeuMFModel()
    activeModel.restoreFromLatestCheckPoint()
    return {'result': 'ok', 'active model': model}

  elif operation == 'train':

    trainingModel = NeuMFModel()
    return trainingModel.train(data['path'], data['rowLimit'], {})

  return 'Triggered operation {} on {} model without specific return value'.format(operation, model)

app.run()
