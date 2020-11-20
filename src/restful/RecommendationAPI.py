from flask_oauthlib.provider import OAuth2Provider
from flask import Flask

app = Flask(__name__)

# TODO oauth2 for strong security
# oauth = OAuth2Provider(app)
app.config["DEBUG"] = True


@app.route('/api/recommendation/<userId>', methods=['GET'])
def getProductRecommendationForUser(userId):
  # Call model & feed the recommendation here
  return [1,2,3,4,5]

@app.route('/api/models', methods=['GET'])
def getSupportedModels():
  # Call model & feed the recommendation here
  return ['MF', 'NeuMF', 'FM', 'NeuFM']

@app.route('/api/models/<operation>/<model>', methods=['POST'])
# operation in [train, active...]
# model in [NeuFM, MF...]
def operateOnModel(operation, model):
  # Call model & feed the recommendation here
  return 'Triggered operation {} on {} model'.format(operation, model)

app.run()
