from flask import Blueprint, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import tempfile
import time
import uuid
from model.process_audio import AudioProcessor
import numpy as np
import librosa
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

process_audio_bp = Blueprint('process_audio', __name__)

# Global processor instance
processor = None

def init_audio_processor():
    """Initialize audio processor with correct model path"""
    global processor
    if processor is None:
        try:
            # Use the correct model path from your requirements
            model_path = "static/model/speech_enhancement_model.pth"
            
            # Ensure model directory exists
            model_dir = os.path.dirname(model_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
                
            processor = AudioProcessor(model_path)
            print(f"✅ Audio processor initialized with model: {model_path}")
            
        except Exception as e:
            print(f"❌ Error initializing audio processor: {e}")
            # Create a fallback processor
            processor = AudioProcessor("dummy_model.pt")  # Will create new model
            
    return processor

@process_audio_bp.route('/process-audio')
def process_audio():
    """Render the audio processing page"""
    return render_template('process_audio.html')

@process_audio_bp.route('/api/upload-audio', methods=['POST'])
def upload_audio():
    """Handle audio file upload and processing"""
    try:
        # Initialize processor
        audio_processor = init_audio_processor()
        
        # Check if file is present
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        allowed_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'Unsupported file format: {file_ext}'}), 400
        
        # Generate unique filenames
        unique_id = str(uuid.uuid4())[:8]
        filename = secure_filename(file.filename)
        base_name = os.path.splitext(filename)[0]
        extension = os.path.splitext(filename)[1]
        
        # Create temp directory for this session
        temp_dir = tempfile.gettempdir()
        session_dir = os.path.join(temp_dir, f"audio_session_{unique_id}")
        os.makedirs(session_dir, exist_ok=True)
        
        # Define file paths
        input_path = os.path.join(session_dir, f"input_{filename}")
        output_filename = f"enhanced_{base_name}_{unique_id}{extension}"
        output_path = os.path.join(session_dir, output_filename)
        
        # Save uploaded file
        file.save(input_path)
        print(f"📁 Saved input file: {input_path}")
        
        # Record processing start time
        start_time = time.time()
        
        # Process audio using the AI model
        print("🔄 Starting audio processing...")
        metrics = audio_processor.enhance_audio(
            input_path=input_path,
            output_path=output_path,
            visualize=False,  # Disable visualization for web processing
            save_plots=False
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        metrics['processing_time_seconds'] = processing_time
        
        # Generate waveform data for visualization
        print("📊 Generating waveform data...")
        original_waveform = generate_waveform_data(input_path)
        enhanced_waveform = generate_waveform_data(output_path)
        
        # Generate comparison waveform image
        waveform_image_path = os.path.join(session_dir, f"waveform_comparison_{unique_id}.png")
        image_generated = generate_waveform_image(input_path, output_path, waveform_image_path)
        
        # Convert numpy types to Python native types for JSON serialization
        serializable_metrics = convert_numpy_types(metrics)
        
        # Add waveform data to response
        response_data = {
            'success': True,
            'metrics': serializable_metrics,
            'output_file': output_filename,
            'session_id': unique_id,
            'waveforms': {
                'original': original_waveform,
                'enhanced': enhanced_waveform
            },
            'waveform_image_available': image_generated
        }
        
        # Clean up input file (keep only output and waveform image)
        if os.path.exists(input_path):
            os.remove(input_path)
        
        print(f"✅ Processing completed in {processing_time:.2f}s")
        print(f"📊 SNR Improvement: {serializable_metrics['snr_improvement_db']:.2f} dB")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error in upload_audio: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    import numpy as np
    
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def generate_waveform_data(audio_path, max_points=2000):
    """Generate waveform data for visualization"""
    try:
        # Load audio file
        waveform, sr = librosa.load(audio_path, sr=None)
        
        # Downsample for visualization if too many points
        if len(waveform) > max_points:
            step = len(waveform) // max_points
            waveform = waveform[::step]
        
        # Create time axis
        time_axis = np.linspace(0, len(waveform) / sr, len(waveform))
        
        return {
            'time': time_axis.tolist(),
            'amplitude': waveform.tolist(),
            'duration': float(len(waveform) / sr),
            'sample_rate': int(sr),
            'max_amplitude': float(np.max(np.abs(waveform)))
        }
    except Exception as e:
        print(f"Error generating waveform data: {e}")
        return None

def generate_waveform_image(original_path, enhanced_path, output_path):
    """Generate comparison waveform image"""
    try:
        # Load both audio files
        original, sr_orig = librosa.load(original_path, sr=None)
        enhanced, sr_enh = librosa.load(enhanced_path, sr=None)
        
        # Create time axes
        time_orig = np.linspace(0, len(original) / sr_orig, len(original))
        time_enh = np.linspace(0, len(enhanced) / sr_enh, len(enhanced))
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot original waveform
        ax1.plot(time_orig, original, color='#ef4444', linewidth=0.5, alpha=0.8)
        ax1.set_title('Original Audio Waveform', fontsize=14, fontweight='bold', color='#ef4444')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#1f2937')
        
        # Plot enhanced waveform
        ax2.plot(time_enh, enhanced, color='#10b981', linewidth=0.5, alpha=0.8)
        ax2.set_title('Enhanced Audio Waveform', fontsize=14, fontweight='bold', color='#10b981')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#1f2937')
        
        # Style the figure
        fig.patch.set_facecolor('#111827')
        for ax in [ax1, ax2]:
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                   facecolor='#111827', edgecolor='none')
        plt.close()
        
        return True
    except Exception as e:
        print(f"Error generating waveform image: {e}")
        return False

@process_audio_bp.route('/api/download-audio/<filename>')
def download_audio(filename):
    """Handle audio file download"""
    try:
        # Extract session ID from filename
        if '_' in filename:
            parts = filename.split('_')
            if len(parts) >= 3:
                session_id = parts[-1].split('.')[0]  # Remove extension
                temp_dir = tempfile.gettempdir()
                session_dir = os.path.join(temp_dir, f"audio_session_{session_id}")
                file_path = os.path.join(session_dir, filename)
                
                if os.path.exists(file_path):
                    print(f"📤 Downloading file: {file_path}")
                    return send_file(
                        file_path, 
                        as_attachment=True,
                        download_name=filename,
                        mimetype='audio/wav'
                    )
        
        # Fallback: check temp directory directly
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, filename)
        
        if os.path.exists(file_path):
            return send_file(
                file_path, 
                as_attachment=True,
                download_name=filename,
                mimetype='audio/wav'
            )
        
        print(f"❌ File not found: {filename}")
        return jsonify({'error': 'File not found'}), 404
        
    except Exception as e:
        print(f"❌ Error in download_audio: {e}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@process_audio_bp.route('/api/batch-process', methods=['POST'])
def batch_process():
    """Handle batch processing of multiple audio files"""
    try:
        audio_processor = init_audio_processor()
        
        files = request.files.getlist('audio_files')
        if not files:
            return jsonify({'error': 'No files provided'}), 400
        
        # Create unique session directory
        unique_id = str(uuid.uuid4())[:8]
        temp_dir = tempfile.gettempdir()
        session_dir = os.path.join(temp_dir, f"batch_session_{unique_id}")
        input_dir = os.path.join(session_dir, 'input')
        output_dir = os.path.join(session_dir, 'output')
        
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all uploaded files
        saved_files = []
        for file in files:
            if file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(input_dir, filename)
                file.save(file_path)
                saved_files.append(filename)
        
        print(f"📁 Processing {len(saved_files)} files in batch...")
        
        # Process batch
        results = audio_processor.batch_process(input_dir, output_dir)
        
        # Convert numpy types and add session info to results
        serializable_results = []
        for result in results:
            result['session_id'] = unique_id
            serializable_results.append(convert_numpy_types(result))
        
        return jsonify({
            'success': True,
            'results': serializable_results,
            'session_id': unique_id,
            'total_files': len(serializable_results)
        })
        
    except Exception as e:
        print(f"❌ Error in batch_process: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Batch processing failed: {str(e)}'}), 500

@process_audio_bp.route('/api/cleanup/<session_id>', methods=['POST'])
def cleanup_session(session_id):
    """Clean up temporary files for a session"""
    try:
        temp_dir = tempfile.gettempdir()
        session_dir = os.path.join(temp_dir, f"audio_session_{session_id}")
        
        if os.path.exists(session_dir):
            import shutil
            shutil.rmtree(session_dir)
            print(f"🧹 Cleaned up session: {session_id}")
            
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"❌ Error cleaning up session {session_id}: {e}")
        return jsonify({'error': str(e)}), 500

@process_audio_bp.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        processor_status = "ready" if processor is not None else "not_initialized"
        
        return jsonify({
            'status': 'healthy',
            'processor': processor_status,
            'timestamp': time.time()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.time()
        }), 500

@process_audio_bp.route('/api/waveform-image/<session_id>')
def get_waveform_image(session_id):
    """Serve waveform comparison image"""
    try:
        temp_dir = tempfile.gettempdir()
        session_dir = os.path.join(temp_dir, f"audio_session_{session_id}")
        image_files = [f for f in os.listdir(session_dir) if f.startswith('waveform_comparison_') and f.endswith('.png')]
        
        if image_files:
            image_path = os.path.join(session_dir, image_files[0])
            if os.path.exists(image_path):
                return send_file(image_path, mimetype='image/png')
        
        return jsonify({'error': 'Waveform image not found'}), 404
        
    except Exception as e:
        print(f"❌ Error serving waveform image: {e}")
        return jsonify({'error': str(e)}), 500

@process_audio_bp.route('/api/realtime-status')
def realtime_status():
    """Get real-time processing status"""
    try:
        from views.realtime_views import init_realtime_processor
        
        realtime_proc = init_realtime_processor()
        stats = realtime_proc.get_realtime_stats()
        
        return jsonify({
            'status': 'available',
            'realtime_stats': stats,
            'timestamp': time.time()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'unavailable',
            'error': str(e),
            'timestamp': time.time()
        }), 500
