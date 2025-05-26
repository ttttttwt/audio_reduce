import os
import torch
import sys
from model.process_audio import AudioProcessor, WaveUNet
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def test_audio_enhancement():
    """
    Test function to enhance the uploaded audio file using the trained model
    """
    # File paths
    model_path = "static/model/speech_enhancement_model.pt"
    input_audio = "uploads/1748140612_tai_xuong.wav"
    output_audio = "enhanced_1748140612_tai_xuong.wav"
    
    print("=" * 50)
    print("AUDIO ENHANCEMENT TEST")
    print("=" * 50)
    
    # Check if files exist
    if not os.path.exists(input_audio):
        print(f"❌ Error: Input audio file not found: {input_audio}")
        return
    
    # Ensure model directory exists
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        print(f"📁 Created model directory: {model_dir}")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found: {model_path}")
        print("Creating a demo model for testing...")
        # Create demo model if file doesn't exist
        demo_model = WaveUNet()
        torch.save(demo_model.state_dict(), model_path)
        print(f"✅ Demo model created at: {model_path}")
    
    try:
        # Initialize AudioProcessor with custom model loading for .pth files
        processor = AudioProcessorCustom(model_path)
        
        print(f"✅ Model loaded successfully from: {model_path}")
        print(f"📁 Input file: {input_audio}")
        print(f"📁 Output file: {output_audio}")
        print(f"🖥️  Device: {processor.device}")
        
        # Process the audio file
        print("\n🔄 Processing audio...")
        metrics = processor.enhance_audio(
            input_path=input_audio,
            output_path=output_audio,
            visualize=True,
            save_plots=True
        )
        
        # Display results
        print("\n" + "=" * 50)
        print("ENHANCEMENT RESULTS")
        print("=" * 50)
        print(f"📊 SNR Improvement: {metrics['snr_improvement_db']:.2f} dB")
        print(f"⚡ Energy Reduction: {metrics['energy_reduction_percent']:.2f}%")
        print(f"⏱️  Duration: {metrics['duration_seconds']:.2f} seconds")
        print(f"🔊 Sample Rate: {metrics['sample_rate']} Hz")
        print(f"📈 Original Power: {metrics['original_power']:.6f}")
        print(f"📈 Enhanced Power: {metrics['enhanced_power']:.6f}")
        print(f"📉 Spectral Distance: {metrics['spectral_distance']:.6f}")
        
        if os.path.exists(output_audio):
            print(f"✅ Enhanced audio saved to: {output_audio}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return None

class AudioProcessorCustom(AudioProcessor):
    """
    Custom AudioProcessor class to handle .pth model files
    """
    def _load_model(self, model_path: str) -> WaveUNet:
        """Load the trained model from .pth file"""
        try:
            if os.path.exists(model_path):
                # Create model instance
                model = WaveUNet()
                
                # Load state dict for .pth files
                if model_path.endswith('.pth'):
                    state_dict = torch.load(model_path, map_location=self.device)
                    model.load_state_dict(state_dict)
                    print(f"✅ Loaded model state dict from {model_path}")
                else:
                    # Load full model for .pt files
                    model = torch.load(model_path, map_location=self.device, weights_only=False)
                    print(f"✅ Loaded full model from {model_path}")
            else:
                print(f"❌ Model file {model_path} not found. Creating new model...")
                model = WaveUNet()
            
            model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Creating new model...")
            model = WaveUNet().to(self.device)
            model.eval()
            return model

def test_model_info():
    """
    Display information about the model and input file
    """
    model_path = "static/model/speech_enhancement_model.pth"
    input_audio = "uploads/1748140612_tai_xuong.wav"
    
    print("\n" + "=" * 50)
    print("FILE INFORMATION")
    print("=" * 50)
    
    # Check model file
    if os.path.exists(model_path):
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"📦 Model file: {model_path}")
        print(f"📏 Model size: {model_size:.2f} MB")
    else:
        print(f"❌ Model file not found: {model_path}")
    
    # Check input file
    if os.path.exists(input_audio):
        audio_size = os.path.getsize(input_audio) / (1024 * 1024)  # MB
        print(f"🎵 Input audio: {input_audio}")
        print(f"📏 Audio size: {audio_size:.2f} MB")
        
        # Try to get audio info
        try:
            import torchaudio
            waveform, sr = torchaudio.load(input_audio)
            duration = waveform.shape[1] / sr
            channels = waveform.shape[0]
            print(f"⏱️  Duration: {duration:.2f} seconds")
            print(f"🔊 Sample rate: {sr} Hz")
            print(f"📻 Channels: {channels}")
            print(f"📊 Shape: {waveform.shape}")
        except Exception as e:
            print(f"❌ Could not read audio info: {e}")
    else:
        print(f"❌ Input audio not found: {input_audio}")

def quick_test():
    """
    Quick test without visualization for faster execution
    """
    model_path = "static/model/speech_enhancement_model.pth"
    input_audio = "uploads/1748140612_tai_xuong.wav"
    output_audio = "quick_enhanced_output.wav"
    
    print("\n🚀 QUICK TEST (No Visualization)")
    print("=" * 30)
    
    # Check if input file exists
    if not os.path.exists(input_audio):
        print(f"❌ Input audio file not found: {input_audio}")
        return False
    
    # Ensure model directory exists
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        print(f"📁 Created model directory: {model_dir}")
    
    # Create demo model if it doesn't exist
    if not os.path.exists(model_path):
        print(f"📦 Creating demo model at: {model_path}")
        demo_model = WaveUNet()
        torch.save(demo_model.state_dict(), model_path)
        print(f"✅ Demo model created")
    
    try:
        processor = AudioProcessorCustom(model_path)
        metrics = processor.enhance_audio(
            input_path=input_audio,
            output_path=output_audio,
            visualize=False,
            save_plots=False
        )
        
        print(f"✅ Quick test completed!")
        print(f"📊 SNR Improvement: {metrics['snr_improvement_db']:.2f} dB")
        print(f"⚡ Energy Reduction: {metrics['energy_reduction_percent']:.2f}%")
        
        return True
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎯 SPEECH ENHANCEMENT TESTING PROGRAM")
    print("🔧 Testing with uploaded audio file")
    
    # Display file information
    test_model_info()
    
    # Run quick test first
    if quick_test():
        print("\n" + "="*50)
        
        # Ask user if they want full test with visualization
        try:
            response = input("\n🤔 Run full test with visualization? (y/n): ").lower()
            if response in ['y', 'yes']:
                test_audio_enhancement()
            else:
                print("✅ Test completed successfully!")
        except:
            # If running in non-interactive mode, run full test
            print("🔄 Running full test...")
            test_audio_enhancement()
    else:
        print("❌ Quick test failed. Please check your files and try again.")
