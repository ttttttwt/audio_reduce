from flask import Blueprint, render_template, request, jsonify
from flask_socketio import emit
import base64
import numpy as np
import torch
import io
from scipy.io import wavfile
from model.process_audio import AudioProcessor
import os

realtime_bp = Blueprint('realtime', __name__)

# Initialize audio processor for real-time processing
realtime_processor = None
# Track active sessions
active_sessions = {}

def init_realtime_processor():
    global realtime_processor
    if realtime_processor is None:
        # Use the same model path as static processing
        model_path = "static/model/speech_enhancement_model.pth"
        
        # Ensure model directory exists
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            
        # Initialize with smaller chunk size for real-time processing
        realtime_processor = AudioProcessor(model_path, chunk_size=2048)
        print(f"✅ Real-time audio processor initialized with model: {model_path}")
    
    return realtime_processor

@realtime_bp.route('/realtime-audio')
def realtime_audio():
    return render_template('realtime_audio.html')

@realtime_bp.route('/api/realtime-status')
def realtime_status():
    """Get real-time processing status"""
    try:
        audio_processor = init_realtime_processor()
        realtime_stats = audio_processor.get_realtime_stats()
        
        return jsonify({
            'status': 'available',
            'realtime_stats': realtime_stats,
            'active_sessions': len(active_sessions)
        })
    except Exception as e:
        return jsonify({
            'status': 'unavailable',
            'error': str(e)
        }), 500

@realtime_bp.route('/api/session-info')
def session_info():
    """Get current session information"""
    user_id = request.args.get('session_id', 'unknown')
    session_data = active_sessions.get(user_id, {})
    
    return jsonify({
        'session_id': user_id,
        'is_active': user_id in active_sessions,
        'session_data': session_data
    })

# Socket event handlers for real-time audio processing
def register_realtime_socket_events(socketio):
    @socketio.on('process_audio_chunk')
    def handle_process_audio_chunk(data):
        try:
            audio_processor = init_realtime_processor()
            user_id = request.sid
            
            # Check if session is active
            if user_id not in active_sessions:
                emit('processing_error', {'error': 'Session not initialized. Please start real-time processing first.'})
                return
            
            # Decode base64 audio data
            audio_data = base64.b64decode(data['audio_data'])
            
            # Convert to numpy array (assuming float32 format from client)
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            
            # Process the audio chunk
            enhanced_audio = audio_processor.process_realtime_chunk(audio_array)
            
            # Convert back to base64
            enhanced_bytes = enhanced_audio.astype(np.float32).tobytes()
            enhanced_b64 = base64.b64encode(enhanced_bytes).decode('utf-8')
            
            # Calculate simple metrics
            original_rms = float(np.sqrt(np.mean(audio_array ** 2)))
            enhanced_rms = float(np.sqrt(np.mean(enhanced_audio ** 2)))
            noise_reduction = float((original_rms - enhanced_rms) / original_rms * 100) if original_rms > 0 else 0
            
            # Update session stats
            if 'chunks_processed' not in active_sessions[user_id]:
                active_sessions[user_id]['chunks_processed'] = 0
            active_sessions[user_id]['chunks_processed'] += 1
            
            emit('audio_processed', {
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
            print(f"❌ Error processing audio chunk: {e}")
            emit('processing_error', {'error': str(e)})

    @socketio.on('start_realtime_processing')
    def handle_start_realtime_processing(data):
        try:
            audio_processor = init_realtime_processor()
            
            user_id = request.sid
            settings = data.get('settings', {})
            
            # Update processor settings if provided
            if 'chunk_size' in settings:
                audio_processor.chunk_size = min(max(settings['chunk_size'], 1024), 8192)
            
            # Initialize session
            active_sessions[user_id] = {
                'settings': settings,
                'start_time': np.datetime64('now'),
                'chunks_processed': 0,
                'status': 'active'
            }
            
            print(f"🎵 User {user_id} started real-time audio processing")
            print(f"⚙️ Settings: {settings}")
            
            emit('realtime_processing_started', {
                'status': 'started',
                'settings': settings,
                'session_id': user_id,
                'model_info': {
                    'model_path': audio_processor.model_path,
                    'device': audio_processor.device,
                    'chunk_size': audio_processor.chunk_size
                }
            })
            
        except Exception as e:
            print(f"❌ Error starting real-time processing: {e}")
            emit('processing_error', {'error': str(e)})

    @socketio.on('stop_realtime_processing')
    def handle_stop_realtime_processing():
        user_id = request.sid
        print(f"🛑 User {user_id} stopped real-time audio processing")
        
        # Clean up session
        if user_id in active_sessions:
            session_data = active_sessions[user_id]
            print(f"📊 Session stats: {session_data.get('chunks_processed', 0)} chunks processed")
            del active_sessions[user_id]
        
        emit('realtime_processing_stopped', {
            'status': 'stopped'
        })

    @socketio.on('update_audio_settings')
    def handle_update_audio_settings(data):
        user_id = request.sid
        settings = data.get('settings', {})
        
        # Update session settings
        if user_id in active_sessions:
            active_sessions[user_id]['settings'].update(settings)
        
        print(f"⚙️ User {user_id} updated audio settings: {settings}")
        
        emit('settings_updated', {
            'status': 'updated',
            'settings': settings
        })

    @socketio.on('get_audio_metrics')
    def handle_get_audio_metrics(data):
        try:
            # Decode audio data
            audio_data = base64.b64decode(data['audio_data'])
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            
            # Calculate comprehensive metrics
            rms = float(np.sqrt(np.mean(audio_array ** 2)))
            peak = float(np.max(np.abs(audio_array)))
            
            # Estimate SNR using percentile method
            signal_power = np.mean(audio_array ** 2)
            noise_floor = np.percentile(np.abs(audio_array), 10) ** 2
            snr_estimate = float(10 * np.log10(signal_power / (noise_floor + 1e-8)))
            
            # Frequency domain analysis
            fft = np.fft.fft(audio_array)
            magnitude_spectrum = np.abs(fft)
            dominant_freq_idx = np.argmax(magnitude_spectrum[:len(magnitude_spectrum)//2])
            sample_rate = data.get('sample_rate', 16000)
            dominant_frequency = float(dominant_freq_idx * sample_rate / len(audio_array))
            
            emit('audio_metrics', {
                'rms': rms,
                'peak': peak,
                'snr_estimate': snr_estimate,
                'dominant_frequency': dominant_frequency,
                'length': len(audio_array),
                'sample_rate': sample_rate
            })
            
        except Exception as e:
            print(f"❌ Error calculating metrics: {e}")
            emit('metrics_error', {'error': str(e)})

    @socketio.on('process_audio_buffer')
    def handle_process_audio_buffer(data):
        """Process larger audio buffers for better quality"""
        try:
            audio_processor = init_realtime_processor()
            user_id = request.sid
            
            # Check if session is active
            if user_id not in active_sessions:
                emit('processing_error', {'error': 'Session not initialized. Please start real-time processing first.'})
                return
            
            # Decode audio data
            audio_data = base64.b64decode(data['audio_data'])
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            
            # Process with overlap-add for better quality
            enhanced_audio = audio_processor.process_buffer_with_overlap(
                audio_array, 
                overlap_ratio=data.get('overlap_ratio', 0.25)
            )
            
            # Convert back to base64
            enhanced_bytes = enhanced_audio.astype(np.float32).tobytes()
            enhanced_b64 = base64.b64encode(enhanced_bytes).decode('utf-8')
            
            emit('buffer_processed', {
                'enhanced_audio': enhanced_b64,
                'processing_info': {
                    'input_length': len(audio_array),
                    'output_length': len(enhanced_audio),
                    'overlap_used': data.get('overlap_ratio', 0.25)
                }
            })
            
        except Exception as e:
            print(f"❌ Error processing audio buffer: {e}")
            emit('processing_error', {'error': str(e)})

    @socketio.on('disconnect')
    def handle_disconnect():
        user_id = request.sid
        if user_id in active_sessions:
            print(f"🔌 User {user_id} disconnected, cleaning up session")
            del active_sessions[user_id]

    @socketio.on('ping_session')
    def handle_ping_session():
        """Keep session alive"""
        user_id = request.sid
        if user_id in active_sessions:
            active_sessions[user_id]['last_ping'] = np.datetime64('now')
            emit('session_pong', {'status': 'alive'})
        else:
            emit('session_pong', {'status': 'inactive'})
