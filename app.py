from flask import Flask
from flask_socketio import SocketIO
from views.video_views import video_bp, register_video_socket_events
from views.process_audio_views import process_audio_bp
from views.realtime_views import realtime_bp, register_realtime_socket_events

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Register blueprints
app.register_blueprint(video_bp)
app.register_blueprint(process_audio_bp)
app.register_blueprint(realtime_bp)

# Register socket event handlers
register_video_socket_events(socketio)
register_realtime_socket_events(socketio)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)