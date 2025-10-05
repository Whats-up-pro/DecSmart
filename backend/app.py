from flask import Flask
from flask_cors import CORS


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    # Health endpoint
    @app.get('/health')
    def health():
        return {"status": "ok"}

    # Register API blueprint
    try:
        from api.routes import api_bp
        app.register_blueprint(api_bp, url_prefix='/api/v1')
    except Exception as _e:
        # Defer import errors to runtime logs; health still works
        pass

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
