"""
Main Flask Application File
Initialize and run the application using application factory pattern
"""

from flask import Flask, jsonify
from flask_cors import CORS
from config import SECRET_KEY, DEBUG, LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT
import logging
from utils.helper import log_info, log_error

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT
)


def create_app(config=None):
    """
    Application factory function
    Creates and configures the Flask application
    """
    try:
        # Create Flask app instance
        app = Flask(__name__)
        
        # Configure app
        app.config['SECRET_KEY'] = SECRET_KEY
        app.config['JSON_SORT_KEYS'] = False
        
        if config:
            app.config.update(config)
        
        # Enable CORS for all routes
        CORS(app, resources={r"/api/*": {"origins": "*"}})
        
        # Register global error handlers
        register_error_handlers(app)
        
        # Register before/after request handlers
        register_request_handlers(app)
        
        # Initialize routes and ML model
        from routes import init_app as routes_init_app
        routes_init_app(app)
        
        log_info("Flask application created successfully")
        log_info("CORS enabled for all API endpoints")
        return app
        
    except Exception as e:
        log_error(f"Error creating Flask application: {str(e)}")
        raise


def register_error_handlers(app):
    """Register global error handlers"""
    
    @app.errorhandler(404)
    def not_found_error(error):
        """Handle 404 errors"""
        log_error(f"404 error: {error}")
        return jsonify({
            'status': 'error',
            'message': 'Endpoint not found',
            'error_code': 404
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors"""
        log_error(f"500 error: {error}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error',
            'error_code': 500
        }), 500

    @app.errorhandler(400)
    def bad_request_error(error):
        """Handle 400 errors"""
        log_error(f"400 error: {error}")
        return jsonify({
            'status': 'error',
            'message': 'Bad request',
            'error_code': 400
        }), 400
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        """Handle 405 errors (Method Not Allowed)"""
        log_error(f"405 error: {error}")
        return jsonify({
            'status': 'error',
            'message': 'Method not allowed',
            'error_code': 405
        }), 405


def register_request_handlers(app):
    """Register before/after request handlers"""
    
    @app.before_request
    def before_request():
        """Execute before each request"""
        pass

    @app.after_request
    def after_request(response):
        """Execute after each request"""
        # Ensure all API responses are JSON with correct content type
        from flask import request
        if request.path.startswith('/api/'):
            response.headers['Content-Type'] = 'application/json'
        return response


# Create application instance
app = create_app()


if __name__ == '__main__':
    log_info("Starting Customer Churn Prediction Application")
    log_info(f"Debug mode: {DEBUG}")
    log_info("Listening on http://127.0.0.1:5000")
    
    try:
        app.run(
            debug=DEBUG,
            host='127.0.0.1',
            port=5000,
            threaded=True,
            use_reloader=DEBUG
        )
    except Exception as e:
        log_error(f"Error running application: {str(e)}")
        raise
        raise
