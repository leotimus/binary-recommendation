import hashlib
import os
import time

from flask import Flask

from src.restful.RecommendationEndpoint import RecommendationEndpoint
from src.restful.oauth2.AuthenticationEndpoint import AuthenticationEndpoint
from src.restful.oauth2.Oauth2 import configOauth2
from src.restful.oauth2.OauthModel import db, User, OAuth2Client


class ApiApp:

  def __init__(self):
    _app = Flask(__name__)
    self._app = _app
    self.app.config.from_json('../../config.json')

    oauthEndpoints = AuthenticationEndpoint().oauthEndpoints
    self.app.register_blueprint(oauthEndpoints)

    endpoints = RecommendationEndpoint().apiEndpoints
    self.app.register_blueprint(endpoints)

    if self.app.config.get('DEV_MODE'):
      self.handleInDevMode()

    db.init_app(self.app)
    configOauth2(self.app)

    _app = self._app

    @_app.before_first_request
    def create_tables():
      if not self.app.config.get('DEV_MODE'):
        return

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

  @property
  def app(self) -> Flask:
      return self._app

  @app.setter
  def app(self, value):
      pass

  def handleInDevMode(self):
    os.environ['AUTHLIB_INSECURE_TRANSPORT'] = '1'
    if os.path.isfile('data/oauth2/db.sqlite'):
      os.remove('data/oauth2/db.sqlite')
