import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.io import wavfile
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=15):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
        )
        self.down = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.conv(x)
        skip = x
        x = self.down(x)
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=15):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose1d(in_channels, in_channels//2, kernel_size=2, stride=2),
            nn.LeakyReLU(0.2)
        )
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class WaveUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DownBlock(1, 16)
        self.down2 = DownBlock(16, 32)
        self.down3 = DownBlock(32, 64)
        self.down4 = DownBlock(64, 128)

        self.bottleneck = nn.Sequential(
            nn.Conv1d(128, 256, 15, padding=7),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        )

        self.up1 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)
        self.up3 = UpBlock(64, 32)
        self.up4 = UpBlock(32, 16)

        self.final = nn.Conv1d(16, 1, 1)

    def forward(self, x):
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)

        x = self.bottleneck(x)

        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        return self.final(x)

class AudioProcessor:
    def __init__(self, model_path: str, device: Optional[str] = None, chunk_size: int = 16384):
        """
        Initialize the Audio Processor
        
        Args:
            model_path: Path to the trained model (.pt or .pth file)
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
            chunk_size: Size of audio chunks for processing
        """
        self.chunk_size = chunk_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Real-time processing state
        self.overlap_buffer = None
        self.processing_history = []
        
        # Video call optimizations
        self.video_call_mode = False
        self.fast_processing_enabled = True
        self.quality_vs_speed_ratio = 0.7  # 0.0 = speed, 1.0 = quality
        
        print(f"✅ Model loaded successfully on {self.device}")
        
    def _load_model(self, model_path: str) -> WaveUNet:
        """Load the trained model from file"""
        try:
            # Create model instance first
            model = WaveUNet()
            
            if os.path.exists(model_path):
                print(f"📦 Loading model from: {model_path}")
                
                if model_path.endswith('.pth'):
                    # Load state dict for .pth files
                    state_dict = torch.load(model_path, map_location=self.device)
                    model.load_state_dict(state_dict)
                    print(f"✅ Loaded model state dict from {model_path}")
                else:
                    # Load full model for .pt files
                    model = torch.load(model_path, map_location=self.device, weights_only=False)
                    print(f"✅ Loaded full model from {model_path}")
            else:
                print(f"⚠️  Model file {model_path} not found. Creating new model...")
                # Ensure model directory exists
                model_dir = os.path.dirname(model_path)
                if model_dir and not os.path.exists(model_dir):
                    os.makedirs(model_dir, exist_ok=True)
                    print(f"📁 Created model directory: {model_dir}")
                
                # Save the new model
                if model_path.endswith('.pth'):
                    torch.save(model.state_dict(), model_path)
                else:
                    torch.save(model, model_path)
                print(f"💾 Saved new model to: {model_path}")
            
            model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔧 Creating fallback model...")
            model = WaveUNet().to(self.device)
            model.eval()
            return model
    
    def preprocess_audio(self, audio_path: str, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
        """
        Preprocess audio file for model input
        
        Args:
            audio_path: Path to input audio file
            target_sr: Target sample rate
            
        Returns:
            Processed audio tensor and sample rate
        """
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
            sr = target_sr
        
        # Normalize amplitude
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        return waveform, sr
    
    def enhance_audio(self, input_path: str, output_path: str, 
                     visualize: bool = True, save_plots: bool = False) -> Dict:
        """
        Enhance an audio file using the pre-trained WaveUNet model
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save enhanced audio
            visualize: Whether to show visualization plots
            save_plots: Whether to save plots to file
            
        Returns:
            Dictionary containing metrics and analysis results
        """
        try:
            print(f"🔄 Processing audio: {input_path}")
            
            # Preprocess audio
            waveform, sr = self.preprocess_audio(input_path)
            original_length = waveform.shape[1]
            
            # Add batch dimension and ensure proper format [batch, channel, length]
            waveform = waveform.unsqueeze(0)
            
            # Pad to multiple of chunk size
            if waveform.shape[2] % self.chunk_size != 0:
                padding = self.chunk_size - (waveform.shape[2] % self.chunk_size)
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            # Process in chunks
            enhanced_chunks = []
            num_chunks = waveform.shape[2] // self.chunk_size
            
            print(f"📊 Processing {num_chunks} chunks of {self.chunk_size} samples each...")
            
            with torch.no_grad():
                for i in range(0, waveform.shape[2], self.chunk_size):
                    chunk = waveform[:, :, i:i+self.chunk_size].to(self.device)
                    
                    # Skip if chunk is too small
                    if chunk.shape[2] < self.chunk_size:
                        # Pad the last chunk
                        padding_needed = self.chunk_size - chunk.shape[2]
                        chunk = torch.nn.functional.pad(chunk, (0, padding_needed))
                    
                    enhanced_chunk = self.model(chunk)
                    enhanced_chunks.append(enhanced_chunk.cpu())
            
            # Combine chunks
            enhanced_waveform = torch.cat(enhanced_chunks, dim=2)
            
            # Remove padding and batch dimension
            enhanced_waveform = enhanced_waveform[:, :, :original_length].squeeze(0)
            original_waveform = waveform[:, :, :original_length].squeeze(0)
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Save enhanced audio
            torchaudio.save(output_path, enhanced_waveform, sr)
            print(f"💾 Enhanced audio saved to: {output_path}")
            
            # Calculate metrics
            metrics = self._calculate_metrics(original_waveform, enhanced_waveform, sr)
            
            # Add file size information (ensure it's a Python float)
            if os.path.exists(output_path):
                output_size = os.path.getsize(output_path)
                metrics['output_file_size_mb'] = float(output_size / (1024 * 1024))
            
            # Visualization (only if requested and not in production)
            if visualize and save_plots:
                plot_save_dir = output_dir if output_dir else "."
                self._visualize_results(original_waveform, enhanced_waveform, sr, 
                                      save_plots, plot_save_dir)
            
            print(f"✅ Processing completed successfully")
            return metrics
            
        except Exception as e:
            print(f"❌ Error in enhance_audio: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def _calculate_metrics(self, noisy: torch.Tensor, enhanced: torch.Tensor, sr: int) -> Dict:
        """Calculate audio quality metrics"""
        noisy_np = noisy.squeeze().numpy()
        enhanced_np = enhanced.squeeze().numpy()
        
        # Signal-to-Noise Ratio improvement
        noisy_power = np.mean(noisy_np ** 2)
        enhanced_power = np.mean(enhanced_np ** 2)
        snr_improvement = 10 * np.log10(enhanced_power / (noisy_power + 1e-8))
        
        # Spectral analysis
        noisy_spec = np.abs(librosa.stft(noisy_np))
        enhanced_spec = np.abs(librosa.stft(enhanced_np))
        
        # Spectral distance
        spectral_distance = np.mean((enhanced_spec - noisy_spec) ** 2)
        
        # Energy reduction (noise reduction indicator)
        energy_reduction = (noisy_power - enhanced_power) / noisy_power * 100
        
        # Convert all numpy types to Python native types
        return {
            'snr_improvement_db': float(snr_improvement),
            'spectral_distance': float(spectral_distance),
            'energy_reduction_percent': float(energy_reduction),
            'sample_rate': int(sr),
            'duration_seconds': float(len(noisy_np) / sr),
            'original_power': float(noisy_power),
            'enhanced_power': float(enhanced_power)
        }
    
    def _visualize_results(self, noisy: torch.Tensor, enhanced: torch.Tensor, sr: int,
                          save_plots: bool = False, save_dir: str = ".") -> None:
        """Create visualization plots"""
        noisy_np = noisy.squeeze().numpy()
        enhanced_np = enhanced.squeeze().numpy()
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # Time domain plots
        time_axis = np.linspace(0, len(noisy_np) / sr, len(noisy_np))
        
        axes[0, 0].plot(time_axis, noisy_np)
        axes[0, 0].set_title('Original Audio Waveform')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(time_axis, enhanced_np)
        axes[0, 1].set_title('Enhanced Audio Waveform')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].grid(True)
        
        # Spectrograms
        noisy_spec = librosa.stft(noisy_np)
        enhanced_spec = librosa.stft(enhanced_np)
        
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(noisy_spec), ref=np.max),
                               y_axis='log', x_axis='time', sr=sr, ax=axes[1, 0])
        axes[1, 0].set_title('Original Audio Spectrogram')
        
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(enhanced_spec), ref=np.max),
                               y_axis='log', x_axis='time', sr=sr, ax=axes[1, 1])
        axes[1, 1].set_title('Enhanced Audio Spectrogram')
        
        # Frequency domain comparison
        noisy_fft = np.abs(np.fft.fft(noisy_np))
        enhanced_fft = np.abs(np.fft.fft(enhanced_np))
        freqs = np.fft.fftfreq(len(noisy_np), 1/sr)
        
        # Only plot positive frequencies
        pos_freqs = freqs[:len(freqs)//2]
        
        axes[2, 0].semilogy(pos_freqs, noisy_fft[:len(freqs)//2], label='Original')
        axes[2, 0].semilogy(pos_freqs, enhanced_fft[:len(freqs)//2], label='Enhanced')
        axes[2, 0].set_title('Frequency Spectrum Comparison')
        axes[2, 0].set_xlabel('Frequency (Hz)')
        axes[2, 0].set_ylabel('Magnitude')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        # Difference spectrum
        diff_fft = enhanced_fft[:len(freqs)//2] - noisy_fft[:len(freqs)//2]
        axes[2, 1].plot(pos_freqs, diff_fft)
        axes[2, 1].set_title('Frequency Difference (Enhanced - Original)')
        axes[2, 1].set_xlabel('Frequency (Hz)')
        axes[2, 1].set_ylabel('Magnitude Difference')
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(save_dir, 'audio_enhancement_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to: {plot_path}")
        
        plt.show()
    
    def batch_process(self, input_dir: str, output_dir: str, 
                     file_extensions: List[str] = ['.wav', '.mp3', '.flac']) -> List[Dict]:
        """
        Process multiple audio files in a directory
        
        Args:
            input_dir: Directory containing input audio files
            output_dir: Directory to save enhanced audio files
            file_extensions: List of audio file extensions to process
            
        Returns:
            List of metrics for each processed file
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        # Find all audio files
        audio_files = []
        for ext in file_extensions:
            audio_files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
        
        print(f"Found {len(audio_files)} audio files to process")
        
        for filename in audio_files:
            try:
                input_path = os.path.join(input_dir, filename)
                output_filename = f"enhanced_{filename}"
                output_path = os.path.join(output_dir, output_filename)
                
                print(f"Processing: {filename}")
                metrics = self.enhance_audio(input_path, output_path, visualize=False)
                metrics['filename'] = filename
                results.append(metrics)
                
                print(f"Completed: {filename} -> {output_filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        return results
    
    def real_time_enhance(self, input_path: str, output_path: str, 
                         window_size: int = 8192, overlap: int = 4096) -> None:
        """
        Simulate real-time audio enhancement with overlapping windows
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save enhanced audio
            window_size: Size of processing window
            overlap: Overlap between windows
        """
        waveform, sr = self.preprocess_audio(input_path)
        enhanced_audio = []
        
        hop_size = window_size - overlap
        
        with torch.no_grad():
            for i in range(0, waveform.shape[1] - window_size + 1, hop_size):
                # Extract window
                window = waveform[:, i:i+window_size].unsqueeze(0).to(self.device)
                
                # Enhance window
                enhanced_window = self.model(window).cpu().squeeze(0)
                
                # Handle overlap
                if i == 0:
                    enhanced_audio.append(enhanced_window)
                else:
                    # Simple overlap-add
                    enhanced_audio.append(enhanced_window[:, overlap:])
        
        # Concatenate all windows
        enhanced_waveform = torch.cat(enhanced_audio, dim=1)
        
        # Save result
        torchaudio.save(output_path, enhanced_waveform, sr)
        print(f"Real-time enhanced audio saved to: {output_path}")

    def enable_video_call_mode(self, enable: bool = True):
        """Enable optimizations for video calls"""
        self.video_call_mode = enable
        if enable:
            # Reduce chunk size for lower latency
            self.chunk_size = min(self.chunk_size, 1024)
            self.fast_processing_enabled = True
            print("🎥 Video call mode enabled - optimized for low latency")
        else:
            print("🎥 Video call mode disabled - optimized for quality")

    def process_realtime_chunk(self, audio_chunk: np.ndarray, target_sr: int = 16000) -> np.ndarray:
        """
        Process a single audio chunk for real-time enhancement
        Optimized for video calls with minimal latency
        
        Args:
            audio_chunk: Input audio chunk as numpy array
            target_sr: Target sample rate
            
        Returns:
            Enhanced audio chunk as numpy array
        """
        try:
            # Fast path for video calls
            if self.video_call_mode and len(audio_chunk) <= 1024:
                return self._fast_process_chunk(audio_chunk)
            
            # Convert to torch tensor
            if isinstance(audio_chunk, np.ndarray):
                audio_tensor = torch.from_numpy(audio_chunk).float()
            else:
                audio_tensor = audio_chunk.float()
            
            # Ensure proper shape [1, 1, length] for model
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
            elif len(audio_tensor.shape) == 2:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Pad to minimum chunk size if needed
            min_chunk_size = 512 if self.video_call_mode else 1024
            original_length = audio_tensor.shape[2]
            
            if audio_tensor.shape[2] < min_chunk_size:
                padding = min_chunk_size - audio_tensor.shape[2]
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
                padded = True
            else:
                padded = False
            
            # Process with model
            with torch.no_grad():
                audio_tensor = audio_tensor.to(self.device)
                
                # Use faster inference for video calls
                if self.video_call_mode:
                    # Skip some processing layers for speed
                    enhanced_tensor = self._fast_inference(audio_tensor)
                else:
                    enhanced_tensor = self.model(audio_tensor)
                
                enhanced_tensor = enhanced_tensor.cpu()
            
            # Remove padding if it was added
            if padded:
                enhanced_tensor = enhanced_tensor[:, :, :original_length]
            
            # Convert back to numpy
            enhanced_audio = enhanced_tensor.squeeze().numpy()
            
            # Ensure output is 1D
            if len(enhanced_audio.shape) > 1:
                enhanced_audio = enhanced_audio.flatten()
            
            # Apply gentle smoothing for video calls to reduce artifacts
            if self.video_call_mode:
                enhanced_audio = self._apply_smoothing(enhanced_audio, audio_chunk)
            
            return enhanced_audio
            
        except Exception as e:
            print(f"❌ Error in process_realtime_chunk: {e}")
            # Return original audio if processing fails
            return audio_chunk if isinstance(audio_chunk, np.ndarray) else audio_chunk.numpy()

    def _fast_process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Ultra-fast processing for small chunks in video calls"""
        try:
            # Simple noise reduction using spectral subtraction
            # This is much faster than full model inference
            
            # Convert to frequency domain
            fft = np.fft.fft(audio_chunk)
            magnitude = np.abs(fft)
            phase = np.angle(fft)
            
            # Estimate noise floor
            noise_floor = np.percentile(magnitude, 20)
            
            # Apply spectral subtraction
            alpha = 2.0  # Over-subtraction factor
            enhanced_magnitude = magnitude - alpha * noise_floor
            enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * magnitude)
            
            # Reconstruct signal
            enhanced_fft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = np.real(np.fft.ifft(enhanced_fft))
            
            return enhanced_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Error in fast processing: {e}")
            return audio_chunk

    def _fast_inference(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """Faster model inference for video calls"""
        try:
            # Use only essential layers for speed
            # This is a simplified version of the full model
            
            # First downsampling layer
            x, skip1 = self.model.down1(audio_tensor)
            x, skip2 = self.model.down2(x)
            
            # Skip some layers for speed
            if self.quality_vs_speed_ratio > 0.5:
                x, skip3 = self.model.down3(x)
                x = self.model.bottleneck(x)
                x = self.model.up1(x, skip3)
            else:
                # Ultra-fast path - minimal processing
                x = self.model.bottleneck(x)
            
            x = self.model.up2(x, skip2)
            x = self.model.up4(x, skip1)
            
            return self.model.final(x)
            
        except Exception as e:
            print(f"❌ Error in fast inference: {e}")
            return self.model(audio_tensor)

    def _apply_smoothing(self, enhanced_audio: np.ndarray, original_audio: np.ndarray) -> np.ndarray:
        """Apply gentle smoothing to reduce artifacts in video calls"""
        try:
            # Use overlap buffer for continuity
            if self.overlap_buffer is not None and len(self.overlap_buffer) > 0:
                # Smooth transition with previous chunk
                transition_length = min(len(enhanced_audio) // 4, len(self.overlap_buffer))
                if transition_length > 0:
                    # Create fade in/out
                    fade_in = np.linspace(0, 1, transition_length)
                    fade_out = np.linspace(1, 0, transition_length)
                    
                    # Apply crossfade
                    enhanced_audio[:transition_length] = (
                        enhanced_audio[:transition_length] * fade_in +
                        self.overlap_buffer[-transition_length:] * fade_out
                    )
            
            # Store overlap buffer for next chunk
            buffer_length = min(len(enhanced_audio) // 4, 64)
            self.overlap_buffer = enhanced_audio[-buffer_length:].copy()
            
            # Apply light noise gate
            threshold = np.percentile(np.abs(enhanced_audio), 10)
            mask = np.abs(enhanced_audio) > threshold
            enhanced_audio = enhanced_audio * mask + original_audio * (1 - mask) * 0.1
            
            return enhanced_audio
            
        except Exception as e:
            print(f"❌ Error in smoothing: {e}")
            return enhanced_audio

    def set_quality_vs_speed(self, ratio: float):
        """Set quality vs speed ratio (0.0 = speed, 1.0 = quality)"""
        self.quality_vs_speed_ratio = max(0.0, min(1.0, ratio))
        print(f"⚙️ Quality vs speed ratio set to: {self.quality_vs_speed_ratio}")

    def reset_realtime_state(self):
        """Reset real-time processing state"""
        self.overlap_buffer = None
        self.processing_history = []
        print("🔄 Real-time processing state reset")

    def get_realtime_stats(self) -> Dict:
        """Get real-time processing statistics"""
        return {
            'device': self.device,
            'chunk_size': self.chunk_size,
            'model_path': self.model_path,
            'has_overlap_buffer': self.overlap_buffer is not None,
            'processing_history_length': len(self.processing_history)
        }

def main():
    """Example usage of the AudioProcessor"""
    # Initialize processor
    model_path = "best_model.pt"  # Path to your trained model
    processor = AudioProcessor(model_path)
    
    # Example single file processing
    input_file = "input_audio.wav"  # Replace with your input file
    output_file = "enhanced_audio.wav"
    
    if os.path.exists(input_file):
        print("Processing single audio file...")
        metrics = processor.enhance_audio(input_file, output_file, visualize=True)
        
        print("\nEnhancement Results:")
        print(f"SNR Improvement: {metrics['snr_improvement_db']:.2f} dB")
        print(f"Energy Reduction: {metrics['energy_reduction_percent']:.2f}%")
        print(f"Duration: {metrics['duration_seconds']:.2f} seconds")
    else:
        print(f"Input file {input_file} not found")
    
    # Example batch processing
    input_dir = "input_audio_dir"
    output_dir = "enhanced_audio_dir"
    
    if os.path.exists(input_dir):
        print(f"\nBatch processing files in {input_dir}...")
        batch_results = processor.batch_process(input_dir, output_dir)
        
        print(f"\nProcessed {len(batch_results)} files")
        for result in batch_results:
            print(f"{result['filename']}: {result['snr_improvement_db']:.2f} dB improvement")

if __name__ == "__main__":
    main()
