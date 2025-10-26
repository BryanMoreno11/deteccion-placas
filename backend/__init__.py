from flask import Flask
from flask_cors import CORS
from backend.routes.plate_routes import bp_plate


def create_app():
    app = Flask(__name__)
    
    # Configurar CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Configuración de upload
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
    app.config['UPLOAD_FOLDER'] = 'uploads'
    
    # Registrar blueprints
    app.register_blueprint(bp_plate, url_prefix="/api")
    
    return app
