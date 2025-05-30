# -*- coding: utf-8 -*-
"""Bản sao của speech-enhancement-wave unet

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JVrsdRsIw0i15CYld3lGdoFHNuvmkQBz
"""

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
import kagglehub
path = kagglehub.dataset_download('jiangwq666/voicebank-demand')

print('Data source import complete.', path)

import kagglehub

# Download latest version
path = kagglehub.dataset_download("jiangwq666/voicebank-demand")

print("Path to dataset files:", path)

"""# Speech Enhancement using U-WaveNet"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/root/.cache/kagglehub/datasets/jiangwq666/voicebank-demand/versions/1'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

"""## Installing dependencies"""

# !pip install torchaudio librosa matplotlib

"""## Imports"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.io import wavfile

"""## Dataset Structure"""

class AudioDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, chunk_size=16384):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.chunk_size = chunk_size
        self.file_list = os.listdir(clean_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        clean_path = os.path.join(self.clean_dir, self.file_list[idx])
        noisy_path = os.path.join(self.noisy_dir, self.file_list[idx])

        # Load audio
        clean_waveform, _ = torchaudio.load(clean_path)
        noisy_waveform, _ = torchaudio.load(noisy_path)

        # Convert to mono
        clean_waveform = clean_waveform.mean(dim=0, keepdim=True)
        noisy_waveform = noisy_waveform.mean(dim=0, keepdim=True)

        # Random chunking
        if clean_waveform.shape[1] >= self.chunk_size:
            start = torch.randint(0, clean_waveform.shape[1] - self.chunk_size, (1,))
            clean_waveform = clean_waveform[:, start:start+self.chunk_size]
            noisy_waveform = noisy_waveform[:, start:start+self.chunk_size]
        else:
            # Pad if too short
            padding = self.chunk_size - clean_waveform.shape[1]
            clean_waveform = torch.nn.functional.pad(clean_waveform, (0, padding))
            noisy_waveform = torch.nn.functional.pad(noisy_waveform, (0, padding))

        return noisy_waveform, clean_waveform

"""## Load Data"""

# Update these paths according to your dataset location
train_clean = "/root/.cache/kagglehub/datasets/jiangwq666/voicebank-demand/versions/1/clean_trainset_28spk_wav"
train_noisy = "/root/.cache/kagglehub/datasets/jiangwq666/voicebank-demand/versions/1/noisy_trainset_28spk_wav"
test_clean = "/root/.cache/kagglehub/datasets/jiangwq666/voicebank-demand/versions/1/clean_testset_wav"
test_noisy = "/root/.cache/kagglehub/datasets/jiangwq666/voicebank-demand/versions/1/noisy_testset_wav"

train_dataset = AudioDataset(train_clean, train_noisy)
test_dataset = AudioDataset(test_clean, test_noisy)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

"""## Model Architecture"""

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

"""## Optimizers & Loss Function"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WaveUNet().to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

"""# Training"""

def train(model, dataloader, epoch):
    model.train()
    total_loss = 0
    progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")

    for noisy, clean in progress:
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()
        outputs = model(noisy)
        loss = criterion(outputs, clean)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)

def validate(model, dataloader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for noisy, clean in dataloader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            outputs = model(noisy)
            loss = criterion(outputs, clean)
            total_loss += loss.item()

    return total_loss / len(dataloader)

best_val_loss = float('inf')
train_losses = []
val_losses = []

num_epochs = 20
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, epoch)
    val_loss = validate(model, test_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # torch.save(model.state_dict(), "best_model.pth")
        torch.save(model, "best_model.pt")
        print("Model saved!")

    scheduler.step(val_loss)

"""## Visual Loss"""

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Training Progress")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

"""## Visual Improvememt"""

def plot_spectrogram(waveform, title):
    waveform = waveform.squeeze().numpy()
    spectrogram = librosa.stft(waveform)
    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram_db, sr=16000, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Load test sample
noisy, clean = next(iter(test_loader))
model = torch.load("best_model.pt")  # Tải toàn bộ mô hình
model.to(device)  # Đảm bảo mô hình được chuyển sang thiết bị (CPU/GPU)
model.eval()
with torch.no_grad():
    enhanced = model(noisy.to(device)).cpu()

# Plot waveforms
plt.figure(figsize=(15, 6))
plt.subplot(3, 1, 1)
plt.plot(noisy[0].squeeze().numpy())
plt.title("Noisy Audio")
plt.subplot(3, 1, 2)
plt.plot(clean[0].squeeze().numpy())
plt.title("Clean Audio")
plt.subplot(3, 1, 3)
plt.plot(enhanced[0].squeeze().numpy())
plt.title("Enhanced Audio")
plt.tight_layout()
plt.show()

"""## Spectograms"""

# Plot spectrograms
plot_spectrogram(noisy[0], "Noisy Spectrogram")
plot_spectrogram(clean[0], "Clean Spectrogram")
plot_spectrogram(enhanced[0], "Enhanced Spectrogram")

"""## Save Model"""

torch.save(model, "speech_enhancement_model.pt")

"""## Use Model"""

def enhance_audio(input_path, output_path):
    """
    Enhance an audio file using the pre-trained WaveUNet model with visualization and analysis
    """
    # Load audio
    waveform, sr = torchaudio.load(input_path)
    waveform = waveform.mean(dim=0, keepdim=True).unsqueeze(0)  # Convert to [1, 1, length] format

    # Pad to multiple of chunk size
    chunk_size = 16384
    if waveform.shape[2] % chunk_size != 0:
        padding = chunk_size - (waveform.shape[2] % chunk_size)
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    # Process in chunks
    enhanced_chunks = []
    for i in range(0, waveform.shape[2], chunk_size):
        chunk = waveform[:, :, i:i+chunk_size].to(device)
        with torch.no_grad():
            enhanced_chunk = model(chunk)
        enhanced_chunks.append(enhanced_chunk.cpu())

    # Combine chunks
    enhanced_waveform = torch.cat(enhanced_chunks, dim=2)

    # Remove any padding
    enhanced_waveform = enhanced_waveform[:, :, :waveform.shape[2]]

    # Save result
    torchaudio.save(output_path, enhanced_waveform.squeeze(0), sr)

    # Convert tensors to numpy arrays for analysis
    noisy_audio = waveform.squeeze().numpy()
    enhanced_audio = enhanced_waveform.squeeze().numpy()

    # Calculate spectrograms
    noisy_spec = librosa.stft(noisy_audio)
    enhanced_spec = librosa.stft(enhanced_audio)

    # Calculate noise metrics
    noise_reduction = 10 * np.log10(np.mean(np.abs(noisy_spec)**2) / np.mean(np.abs(enhanced_spec)**2))

    # Plot waveforms and spectrograms
    plt.figure(figsize=(15, 10))

    # Plot waveforms
    plt.subplot(4, 1, 1)
    plt.plot(noisy_audio)
    plt.title('Noisy Audio Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.subplot(4, 1, 2)
    plt.plot(enhanced_audio)
    plt.title('Enhanced Audio Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    # Plot spectrograms
    plt.subplot(4, 1, 3)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(noisy_spec), ref=np.max),
                           y_axis='log', x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Noisy Audio Spectrogram')

    plt.subplot(4, 1, 4)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(enhanced_spec), ref=np.max),
                           y_axis='log', x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Enhanced Audio Spectrogram')

    plt.tight_layout()
    plt.show()

    # Print noise reduction metrics
    print(f"Noise Reduction: {noise_reduction:.2f} dB")

    # Play audio using IPython display
    from IPython.display import Audio, display

    print("\nNoisy Audio:")
    display(Audio(noisy_audio, rate=sr))

    print("\nEnhanced Audio:")
    display(Audio(enhanced_audio, rate=sr))

    return {
        'noise_reduction': noise_reduction,
        'sample_rate': sr,
        'noisy_duration': len(noisy_audio) / sr,
        'enhanced_duration': len(enhanced_audio) / sr
    }

# Usage:
# metrics = enhance_audio("/kaggle/input/voicebank-demand/noisy_trainset_28spk_wav/p226_014.wav",
#                        "/kaggle/working/output_clean.wav")

import os

# Create output directory if it doesn't exist
os.makedirs("/kaggle/working/", exist_ok=True)

metrics = enhance_audio("/kaggle/input/voicebank-demand/noisy_trainset_28spk_wav/p226_014.wav",
                       "/kaggle/working/output_clean.wav")
print("\nAnalysis Results:")
print(f"Noise Reduction: {metrics['noise_reduction']:.2f} dB")
print(f"Audio Duration: {metrics['noisy_duration']:.2f} seconds")
print(f"Sample Rate: {metrics['sample_rate']} Hz")