from flask import Blueprint, render_template, request
from flask_socketio import emit, join_room, leave_room
import uuid
import base64
import numpy as np
import os
from model.process_audio import AudioProcessor

video_bp = Blueprint('video', __name__)

# Store room and user information
rooms = {}
users = {}

# Initialize audio processor for video calls
video_audio_processor = None

def init_video_audio_processor():
    global video_audio_processor
    if video_audio_processor is None:
        model_path = "static/model/speech_enhancement_model.pth"
        
        # Ensure model directory exists
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            
        # Initialize with smaller chunk size for real-time processing
        video_audio_processor = AudioProcessor(model_path, chunk_size=16384)
        print(f"✅ Video call audio processor initialized with model: {model_path}")
    
    return video_audio_processor

@video_bp.route('/')
def index():
    return render_template('index.html')

@video_bp.route('/video')
def video_index():
    return render_template('index_video.html')

@video_bp.route('/room/<room_id>')
def room(room_id):
    return render_template('test.html', room_id=room_id)

@video_bp.route('/join')
def join():
    return render_template('join.html')

@video_bp.route('/lobby/<room_id>')
def lobby(room_id):
    username = request.args.get('username', 'Anonymous')
    return render_template('lobby.html', room_id=room_id, username=username)

# Socket event handlers for video functionality
def register_video_socket_events(socketio):
    @socketio.on('create_room')
    def handle_create_room(data):
        room_id = str(uuid.uuid4())[:8]
        username = data.get('username', 'Anonymous')
        
        rooms[room_id] = {
            'participants': {},
            'messages': [],
            'audio_processing_enabled': {}  # Track AI processing per user
        }
        
        emit('room_created', {'room_id': room_id, 'username': username})

    @socketio.on('join_room')
    def handle_join_room(data):
        room_id = data.get('room_id')
        username = data.get('username', 'Anonymous')
        initial_video_enabled = data.get('video_enabled', True)
        initial_audio_enabled = data.get('audio_enabled', True)
        
        if room_id not in rooms:
            emit('error', {'message': 'Phòng không tồn tại'})
            return
        
        user_id = request.sid
        users[user_id] = {
            'username': username,
            'room_id': room_id,
            'video_enabled': initial_video_enabled,
            'audio_enabled': initial_audio_enabled,
            'user_id': user_id,
            'noise_reduction_enabled': False  # Track noise reduction state
        }
        
        rooms[room_id]['participants'][user_id] = users[user_id]
        rooms[room_id]['audio_processing_enabled'][user_id] = False
        join_room(room_id)
        
        print(f"User {username} ({user_id}) joined room {room_id}")
        print(f"Current participants in room {room_id}: {len(rooms[room_id]['participants'])}")
        
        current_participants = []
        for participant_id, participant_data in rooms[room_id]['participants'].items():
            current_participants.append({
                'user_id': participant_id,
                'username': participant_data['username'],
                'video_enabled': participant_data.get('video_enabled', True),
                'audio_enabled': participant_data.get('audio_enabled', True)
            })
        
        emit('user_joined', {
            'user_id': user_id,
            'username': username,
            'participants': current_participants
        })
        
        emit('user_joined', {
            'user_id': user_id,
            'username': username,
            'participants': current_participants
        }, room=room_id, include_self=False)
        
        emit('chat_history', {'messages': rooms[room_id]['messages']})

    @socketio.on('leave_room')
    def handle_leave_room():
        user_id = request.sid
        if user_id in users:
            room_id = users[user_id]['room_id']
            username = users[user_id]['username']
            
            print(f"User {username} ({user_id}) leaving room {room_id}")
            
            if room_id in rooms and user_id in rooms[room_id]['participants']:
                del rooms[room_id]['participants'][user_id]
            
            del users[user_id]
            leave_room(room_id)
            
            print(f"Remaining participants in room {room_id}: {len(rooms[room_id]['participants']) if room_id in rooms else 0}")
            
            remaining_participants = []
            if room_id in rooms:
                for participant_id, participant_data in rooms[room_id]['participants'].items():
                    remaining_participants.append({
                        'user_id': participant_id,
                        'username': participant_data['username'],
                        'video_enabled': participant_data.get('video_enabled', True),
                        'audio_enabled': participant_data.get('audio_enabled', True)
                    })
            
            emit('user_left', {
                'user_id': user_id,
                'username': username,
                'participants': remaining_participants
            }, room=room_id)
            
            if room_id in rooms and len(rooms[room_id]['participants']) == 0:
                print(f"Room {room_id} is empty, removing it")
                del rooms[room_id]

    @socketio.on('disconnect')
    def handle_disconnect():
        handle_leave_room()

    # WebRTC Signaling
    @socketio.on('offer')
    def handle_offer(data):
        target_id = data.get('target')
        emit('offer', data, room=target_id)

    @socketio.on('answer')
    def handle_answer(data):
        target_id = data.get('target')
        emit('answer', data, room=target_id)

    @socketio.on('ice_candidate')
    def handle_ice_candidate(data):
        target_id = data.get('target')
        emit('ice_candidate', data, room=target_id)

    # Chat
    @socketio.on('send_message')
    def handle_send_message(data):
        user_id = request.sid
        if user_id in users:
            room_id = users[user_id]['room_id']
            username = users[user_id]['username']
            message = data.get('message', '')
            
            message_data = {
                'user_id': user_id,
                'username': username,
                'message': message,
                'timestamp': data.get('timestamp')
            }
            
            rooms[room_id]['messages'].append(message_data)
            emit('new_message', message_data, room=room_id)

    # Media Controls
    @socketio.on('toggle_video')
    def handle_toggle_video(data):
        user_id = request.sid
        if user_id in users:
            room_id = users[user_id]['room_id']
            enabled = data.get('enabled', False)
            
            users[user_id]['video_enabled'] = enabled
            rooms[room_id]['participants'][user_id]['video_enabled'] = enabled
            
            emit('user_video_toggle', {
                'user_id': user_id,
                'enabled': enabled
            }, room=room_id)

    @socketio.on('toggle_audio')
    def handle_toggle_audio(data):
        user_id = request.sid
        if user_id in users:
            room_id = users[user_id]['room_id']
            enabled = data.get('enabled', False)
            
            users[user_id]['audio_enabled'] = enabled
            rooms[room_id]['participants'][user_id]['audio_enabled'] = enabled
            
            emit('user_audio_toggle', {
                'user_id': user_id,
                'enabled': enabled
            }, room=room_id)

    # Screen Sharing
    @socketio.on('start_screen_sharing')
    def handle_start_screen_sharing():
        user_id = request.sid
        if user_id in users:
            room_id = users[user_id]['room_id']
            username = users[user_id]['username']
            
            users[user_id]['sharing_screen'] = True
            rooms[room_id]['participants'][user_id]['sharing_screen'] = True
            
            emit('user_screen_sharing', {
                'user_id': user_id,
                'username': username,
                'sharing': True
            }, room=room_id)

    @socketio.on('stop_screen_sharing')
    def handle_stop_screen_sharing():
        user_id = request.sid
        if user_id in users:
            room_id = users[user_id]['room_id']
            username = users[user_id]['username']
            
            users[user_id]['sharing_screen'] = False
            rooms[room_id]['participants'][user_id]['sharing_screen'] = False
            
            emit('user_screen_sharing', {
                'user_id': user_id,
                'username': username,
                'sharing': False
            }, room=room_id)

    # AI Audio Processing Events
    @socketio.on('start_video_noise_reduction')
    def handle_start_video_noise_reduction(data):
        try:
            audio_processor = init_video_audio_processor()
            user_id = request.sid
            
            if user_id not in users:
                emit('noise_reduction_error', {'error': 'User not found'})
                return
                
            room_id = users[user_id]['room_id']
            settings = data.get('settings', {})
            
            # Update processor settings if provided
            if 'chunk_size' in settings:
                audio_processor.chunk_size = min(max(settings['chunk_size'], 1024), 8192)
            
            # Update user state
            users[user_id]['noise_reduction_enabled'] = True
            rooms[room_id]['audio_processing_enabled'][user_id] = True
            
            print(f"🎵 User {users[user_id]['username']} started noise reduction in room {room_id}")
            
            emit('video_noise_reduction_started', {
                'status': 'started',
                'settings': settings,
                'model_info': {
                    'model_path': audio_processor.model_path,
                    'device': str(audio_processor.device),
                    'chunk_size': audio_processor.chunk_size
                }
            })
            
            # Notify other participants
            emit('user_noise_reduction_toggle', {
                'user_id': user_id,
                'username': users[user_id]['username'],
                'enabled': True
            }, room=room_id, include_self=False)
            
        except Exception as e:
            print(f"❌ Error starting video noise reduction: {e}")
            emit('noise_reduction_error', {'error': str(e)})

    @socketio.on('stop_video_noise_reduction')
    def handle_stop_video_noise_reduction():
        user_id = request.sid
        
        if user_id not in users:
            return
            
        room_id = users[user_id]['room_id']
        username = users[user_id]['username']
        
        # Update user state
        users[user_id]['noise_reduction_enabled'] = False
        rooms[room_id]['audio_processing_enabled'][user_id] = False
        
        print(f"🛑 User {username} stopped noise reduction in room {room_id}")
        
        emit('video_noise_reduction_stopped', {'status': 'stopped'})
        
        # Notify other participants
        emit('user_noise_reduction_toggle', {
            'user_id': user_id,
            'username': username,
            'enabled': False
        }, room=room_id, include_self=False)

    @socketio.on('process_video_audio_chunk')
    def handle_process_video_audio_chunk(data):
        try:
            audio_processor = init_video_audio_processor()
            user_id = request.sid
            
            # Check if user has noise reduction enabled
            if user_id not in users or not users[user_id].get('noise_reduction_enabled', False):
                emit('noise_reduction_error', {'error': 'Noise reduction not enabled'})
                return
            
            # Decode base64 audio data
            audio_data = base64.b64decode(data['audio_data'])
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            
            # Process the audio chunk
            enhanced_audio = audio_processor.process_realtime_chunk(audio_array)
            
            # Convert back to base64
            enhanced_bytes = enhanced_audio.astype(np.float32).tobytes()
            enhanced_b64 = base64.b64encode(enhanced_bytes).decode('utf-8')
            
            # Calculate metrics
            original_rms = float(np.sqrt(np.mean(audio_array ** 2)))
            enhanced_rms = float(np.sqrt(np.mean(enhanced_audio ** 2)))
            noise_reduction = float((original_rms - enhanced_rms) / original_rms * 100) if original_rms > 0 else 0
            
            emit('video_audio_processed', {
                'enhanced_audio': enhanced_b64,
                'original_length': len(audio_array),
                'enhanced_length': len(enhanced_audio),
                'metrics': {
                    'original_rms': original_rms,
                    'enhanced_rms': enhanced_rms,
                    'noise_reduction_percent': noise_reduction
                }
            })
            
        except Exception as e:
            print(f"❌ Error processing video audio chunk: {e}")
            emit('noise_reduction_error', {'error': str(e)})

    @socketio.on('get_video_audio_status')
    def handle_get_video_audio_status():
        try:
            audio_processor = init_video_audio_processor()
            user_id = request.sid
            
            if user_id in users:
                room_id = users[user_id]['room_id']
                processing_users = []
                
                for uid, enabled in rooms[room_id]['audio_processing_enabled'].items():
                    if enabled and uid in users:
                        processing_users.append({
                            'user_id': uid,
                            'username': users[uid]['username']
                        })
                
                emit('video_audio_status', {
                    'processor_available': True,
                    'device': str(audio_processor.device),
                    'users_with_processing': processing_users,
                    'current_user_enabled': users[user_id].get('noise_reduction_enabled', False)
                })
            else:
                emit('video_audio_status', {
                    'processor_available': False,
                    'error': 'User not found'
                })
                
        except Exception as e:
            emit('video_audio_status', {
                'processor_available': False,
                'error': str(e)
            })
