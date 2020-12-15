from authlib.integrations.flask_oauth2 import (
  AuthorizationServer,
  ResourceProtector,
)
from authlib.integrations.sqla_oauth2 import (
  create_query_client_func,
  create_revocation_endpoint,
  create_bearer_token_validator,
)
from authlib.oauth2.rfc6749 import grants
from .OauthModel import db, User
from .OauthModel import OAuth2Client, OAuth2Token


class PasswordGrant(grants.ResourceOwnerPasswordCredentialsGrant):
  def authenticate_user(self, username, password):
    user = User.query.filter_by(username=username).first()
    if user is not None and user.validate(password):
      return user


class RefreshTokenGrant(grants.RefreshTokenGrant):
  def authenticate_refresh_token(self, refresh_token):
    token = OAuth2Token.query.filter_by(refresh_token=refresh_token).first()
    if token and token.is_refresh_token_active():
      return token

  def authenticate_user(self, credential):
    return User.query.get(credential.user_id)

  def revoke_old_credential(self, credential):
    credential.revoked = True
    db.session.add(credential)
    db.session.commit()


def createSaveTokenFunc(session, token_model):
  def save_token(token, request):
    if request.user:
      user_id = request.user.get_user_id()
    else:
      user_id = None
    client = request.client
    item = token_model(
      client_id=client.client_id,
      user_id=user_id,
      scope=client.scope,
      **token
    )
    session.add(item)
    session.commit()

  return save_token


def configOauth2(app):
  authorization.init_app(app)

  # support all grants
  authorization.register_grant(grants.ClientCredentialsGrant)
  authorization.register_grant(PasswordGrant)
  authorization.register_grant(RefreshTokenGrant)

  # support revocation
  revocation_cls = create_revocation_endpoint(db.session, OAuth2Token)
  authorization.register_endpoint(revocation_cls)

  # protect resource
  bearer_cls = create_bearer_token_validator(db.session, OAuth2Token)
  require_oauth.register_token_validator(bearer_cls())


query_client = create_query_client_func(db.session, OAuth2Client)
save_token = createSaveTokenFunc(db.session, OAuth2Token)
authorization = AuthorizationServer(
  query_client=query_client,
  save_token=save_token,
)
require_oauth = ResourceProtector()
