import hashlib
import os
import time

from flask import Flask
from flask.logging import default_handler

from src.logger import Logger
from src.restful.RecommendationEndpoint import RecommendationEndpoint
from src.restful.oauth2.AuthenticationEndpoint import AuthenticationEndpoint
from src.restful.oauth2.Oauth2 import configOauth2
from src.restful.oauth2.OauthModel import db, User, OAuth2Client


class ApiApp:

  def __init__(self):
    self._logger = Logger.getLogger(__name__, logPath='logs/server.log', console=True)
    self._app: Flask = Flask(__name__)
    self._app.logger.removeHandler(default_handler)
    self._app.logger.addHandler(self._logger.handlers[1])
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

      self.logger.info("In dev mode, auto create authentication database.")
      db.create_all()

      self.logger.info("In dev mode, create default admin.")
      admin = User(username=self.app.config.get('DEF_ADMIN'),
                   password=hashlib.md5(self.app.config.get('DEF_ADMIN_PASS').encode()).hexdigest(),
                   scope='manager')
      db.session.add(admin)

      self.logger.info("In dev mode, create default client.")
      client_id = self.app.config.get('DEV_CLIENT_ID')
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
      adminClient.client_secret = self.app.config.get('DEV_CLIENT_SECRET')
      adminClient.set_client_metadata(client_metadata)
      db.session.add(adminClient)

      db.session.commit()

  @property
  def logger(self) -> Logger:
      return self._logger

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
