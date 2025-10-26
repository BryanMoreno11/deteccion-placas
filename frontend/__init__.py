from flask import Flask
from frontend.gestionPlacas.routes import bp_gestion_placas


def create_app():
    app = Flask(__name__)
    
    # Registrar blueprints
    app.register_blueprint(bp_gestion_placas)
    
    return app
