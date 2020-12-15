import time

from flask import Blueprint, request, session
from werkzeug.security import gen_salt

from src.restful.oauth2.Oauth2 import authorization
from src.restful.oauth2.OauthModel import OAuth2Client, db, User


class AuthenticationEndpoint:

  def __init__(self):
    _oauthEndpoints = Blueprint(__name__, 'oauth2')
    self._oauthEndpoints = _oauthEndpoints

    # OAUTH2 #
    @_oauthEndpoints.route('/create_client', methods=(['POST']))
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

    @_oauthEndpoints.route('/oauth/token', methods=['POST'])
    def issue_token():
      response = authorization.create_token_response()
      return response

    @_oauthEndpoints.route('/oauth/revoke', methods=['POST'])
    def revoke_token():
      return authorization.create_endpoint_response('revocation')

    def splitByCrlf(s):
      return [v for v in s.splitlines() if v]

    def currentUser():
      if 'id' in session:
        uid = session['id']
        return User.query.get(uid)
      return None

  @property
  def oauthEndpoints(self):
      return self._oauthEndpoints

  @oauthEndpoints.setter
  def oauthEndpoints(self, value):
      pass
