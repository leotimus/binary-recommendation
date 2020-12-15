from flask import Flask

from src.restful.ApiApp import ApiApp


apiApp = ApiApp()
app: Flask
app = apiApp.app
app.run()
