
from flask import jsonify, request, Blueprint
from flask_oauthlib.provider import OAuth2Provider

from src.models import RModel
from src.models.NCFModel import NCFModel
from src.models.NeuMFModel import NeuMFModel
from src.restful.oauth2.Oauth2 import require_oauth


class RecommendationEndpoint:

  def __init__(self):
    self._activeModel = None
    _apiEndpoints = Blueprint(__name__, 'api')
    self._apiEndpoints = _apiEndpoints
    oauth = OAuth2Provider(self._apiEndpoints)

    @_apiEndpoints.route('/api/recommendation/<int:userId>/<int:numberOfItem>', methods=['GET'])
    def getProductRecommendationForUser(userId, numberOfItem):
      # Call model & feed the recommendation here
      global activeModel
      return jsonify(activeModel.predictForUser(userId, numberOfItem))

    @_apiEndpoints.route('/api/users', methods=['GET'])
    @require_oauth('manager')
    def getUsers():
      # Call model & feed the recommendation here
      global activeModel
      return jsonify(activeModel.getPredictableUsers())

    @_apiEndpoints.route('/api/models', methods=['GET'])
    @require_oauth('manager')
    def getSupportedModels():
      # Call model & feed the recommendation here
      return str('NeuMFModel, NCFModel')

    # operation in [train, active...]
    # model in [NeuFM, MF...]
    @_apiEndpoints.route('/api/models/<operation>/<model>', methods=['POST'])
    @require_oauth('manager')
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

  @property
  def activeModel(self) -> RModel:
      return self._activeModel

  @activeModel.setter
  def activeModel(self, value):
      self._activeModel = value

  @property
  def apiEndpoints(self) -> Blueprint:
      return self._apiEndpoints

  @apiEndpoints.setter
  def apiEndpoints(self, value):
      pass
