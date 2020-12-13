import hashlib, os, time

from flask_oauthlib.provider import OAuth2Provider
from flask import Flask, jsonify, request, session
from werkzeug.security import gen_salt

from src.models import RModel
from src.models.NCFModel import NCFModel
from src.models.NeuMFModel import NeuMFModel
from src.restful.oauth2.Oauth2 import configOauth2, authorization, require_oauth
from src.restful.oauth2.OauthModel import db, User, OAuth2Client

#TODO fix me to dev mode
os.environ['AUTHLIB_INSECURE_TRANSPORT'] = '1'
activeModel: RModel

app = Flask(__name__)
app.config["DEBUG"] = True
app.config['OAUTH2_REFRESH_TOKEN_GENERATOR'] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
#TODO fix me to dev mode and init db properly
os.remove('src/restful/db.sqlite')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'

oauth = OAuth2Provider(app)


@app.before_first_request
def create_tables():
  db.create_all()

  admin = User(username='admin', password=hashlib.md5('12345'.encode()).hexdigest(), scope='manager')
  db.session.add(admin)

  client_id = 'lkchCRIIOR3xi7qFJI6zPPEH'
  client_id_issued_at = int(time.time())
  adminClient = OAuth2Client(client_id=client_id, client_id_issued_at=client_id_issued_at, user_id=admin.id)
  client_metadata = {
    "client_name": 'sample_client_name',
    "client_uri": 'sample_client_uri',
    "scope": 'manager',
    "grant_types": ['authorization_code', 'password'],
    "redirect_uris": 'www.google.com',
    "response_types": 'code',
    "token_endpoint_auth_method": 'client_secret_basic'
  }
  adminClient.client_secret = '8WUn6UXefjNVsNP6Q6zz3opeIgBh3wLzAO5Kx0ZUQWlXXV2d'
  adminClient.set_client_metadata(client_metadata)
  db.session.add(adminClient)

  db.session.commit()


db.init_app(app)
configOauth2(app)

# OAUTH2 #
@app.route('/create_client', methods=(['POST']))
def create_client():
  user = currentUser()
  if not user:
    return 'Not authenticated', 403

  client_id = gen_salt(24)
  client_id_issued_at = int(time.time())
  client = OAuth2Client(client_id=client_id, client_id_issued_at=client_id_issued_at, user_id=user.id)

  form = request.form
  client_metadata = {
    "client_name": form["client_name"],
    "client_uri": form["client_uri"],
    "grant_types": splitByCrlf(form["grant_type"]),
    "redirect_uris": splitByCrlf(form["redirect_uri"]),
    "response_types": splitByCrlf(form["response_type"]),
    "scope": form["scope"],
    "token_endpoint_auth_method": form["token_endpoint_auth_method"]
  }
  client.set_client_metadata(client_metadata)

  if form['token_endpoint_auth_method'] == 'none':
    client.client_secret = ''
  else:
    client.client_secret = gen_salt(48)

  db.session.add(client)
  db.session.commit()
  return 'OK', 200


@app.route('/oauth/authorize', methods=['POST'])
def authorize():
  user = currentUser()
  # if user log status is not true (Auth server), then to log it in
  if not user:
    return 'Authorized', 200
  if not user and 'username' in request.form:
    username = request.form.get('username')
    user = User.query.filter_by(username=username).first()
  if request.form['confirm']:
    grantUser = user
  else:
    grantUser = None
  return authorization.create_authorization_response(grant_user=grantUser)


@app.route('/oauth/token', methods=['POST'])
def issue_token():
  response = authorization.create_token_response()
  return response


@app.route('/oauth/revoke', methods=['POST'])
def revoke_token():
  return authorization.create_endpoint_response('revocation')


# MAIN API
@app.route('/api/recommendation/<int:userId>/<int:numberOfItem>', methods=['GET'])
def getProductRecommendationForUser(userId, numberOfItem):
  # Call model & feed the recommendation here
  global activeModel
  return jsonify(activeModel.predictForUser(userId, numberOfItem))


@app.route('/api/users', methods=['GET'])
@require_oauth('manager')
def getUsers():
  # Call model & feed the recommendation here
  global activeModel
  return jsonify(activeModel.getPredictableUsers())


@app.route('/api/models', methods=['GET'])
@require_oauth('manager')
def getSupportedModels():
  # Call model & feed the recommendation here
  return str('NeuMFModel, NCFModel')


# operation in [train, active...]
# model in [NeuFM, MF...]
@app.route('/api/models/<operation>/<model>', methods=['POST'])
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


def currentUser():
  if 'id' in session:
    uid = session['id']
    return User.query.get(uid)
  return None


def splitByCrlf(s):
  return [v for v in s.splitlines() if v]


app.run()
