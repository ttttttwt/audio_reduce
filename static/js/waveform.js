class WaveformVisualizer {
    constructor() {
        this.originalCanvas = document.getElementById('original-waveform-canvas');
        this.enhancedCanvas = document.getElementById('enhanced-waveform-canvas');
        this.originalCtx = this.originalCanvas?.getContext('2d');
        this.enhancedCtx = this.enhancedCanvas?.getContext('2d');
        this.currentSessionId = null;
        this.waveformData = null;
        this.showImage = true;
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // Toggle between image and canvas view
        const toggleBtn = document.getElementById('toggle-waveform-view');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => this.toggleWaveformView());
        }

        // Download waveform image
        const downloadBtn = document.getElementById('download-waveform-image');
        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => this.downloadWaveformImage());
        }
    }

    displayWaveforms(sessionId, waveformData, hasImage = false) {
        this.currentSessionId = sessionId;
        this.waveformData = waveformData;

        // Show waveform section
        const waveformSection = document.querySelector('.backdrop-blur-lg.bg-white\\/10:has(#waveform-image-container)');
        if (waveformSection) {
            waveformSection.classList.remove('hidden');
        }

        // Display image if available
        if (hasImage) {
            this.loadWaveformImage(sessionId);
        }

        // Draw canvas waveforms
        this.drawWaveforms();
        
        // Update duration and amplitude info
        this.updateWaveformInfo();
    }

    loadWaveformImage(sessionId) {
        const imageContainer = document.getElementById('waveform-image-container');
        const image = document.getElementById('waveform-comparison-image');
        const loading = document.getElementById('waveform-loading');

        if (!imageContainer || !image || !loading) return;

        // Show loading
        loading.classList.remove('hidden');
        image.classList.add('hidden');

        // Load image
        const imageUrl = `/api/waveform-image/${sessionId}?t=${Date.now()}`;
        image.onload = () => {
            loading.classList.add('hidden');
            image.classList.remove('hidden');
        };
        image.onerror = () => {
            loading.classList.add('hidden');
            console.error('Failed to load waveform image');
        };
        image.src = imageUrl;
    }

    drawWaveforms() {
        if (!this.waveformData || !this.originalCtx || !this.enhancedCtx) return;

        // Draw original waveform
        this.drawWaveform(
            this.originalCtx, 
            this.waveformData.original, 
            '#ef4444' // Red color
        );

        // Draw enhanced waveform
        this.drawWaveform(
            this.enhancedCtx, 
            this.waveformData.enhanced, 
            '#10b981' // Green color
        );
    }

    drawWaveform(ctx, data, color) {
        if (!data || !data.amplitude || !data.time) return;

        const canvas = ctx.canvas;
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.fillStyle = '#111827';
        ctx.fillRect(0, 0, width, height);

        // Draw grid
        this.drawGrid(ctx, width, height);

        // Prepare waveform data
        const amplitudes = data.amplitude;
        const maxAmplitude = Math.max(...amplitudes.map(Math.abs));
        
        if (maxAmplitude === 0) return;

        // Draw waveform
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.8;

        ctx.beginPath();
        for (let i = 0; i < amplitudes.length; i++) {
            const x = (i / (amplitudes.length - 1)) * width;
            const normalizedAmp = amplitudes[i] / maxAmplitude;
            const y = height / 2 - (normalizedAmp * height * 0.4);

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();

        // Draw center line
        ctx.strokeStyle = '#374151';
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.3;
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();

        ctx.globalAlpha = 1;
    }

    drawGrid(ctx, width, height) {
        ctx.strokeStyle = '#374151';
        ctx.lineWidth = 0.5;
        ctx.globalAlpha = 0.2;

        // Vertical grid lines
        for (let i = 0; i <= 10; i++) {
            const x = (i / 10) * width;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }

        // Horizontal grid lines
        for (let i = 0; i <= 4; i++) {
            const y = (i / 4) * height;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }

        ctx.globalAlpha = 1;
    }

    updateWaveformInfo() {
        if (!this.waveformData) return;

        // Update original audio info
        const originalDuration = document.getElementById('original-duration');
        const originalMaxAmp = document.getElementById('original-max-amplitude');
        
        if (originalDuration && this.waveformData.original) {
            originalDuration.textContent = `Duration: ${this.waveformData.original.duration.toFixed(2)}s`;
        }
        if (originalMaxAmp && this.waveformData.original) {
            originalMaxAmp.textContent = `Max: ${this.waveformData.original.max_amplitude.toFixed(3)}`;
        }

        // Update enhanced audio info
        const enhancedDuration = document.getElementById('enhanced-duration');
        const enhancedMaxAmp = document.getElementById('enhanced-max-amplitude');
        
        if (enhancedDuration && this.waveformData.enhanced) {
            enhancedDuration.textContent = `Duration: ${this.waveformData.enhanced.duration.toFixed(2)}s`;
        }
        if (enhancedMaxAmp && this.waveformData.enhanced) {
            enhancedMaxAmp.textContent = `Max: ${this.waveformData.enhanced.max_amplitude.toFixed(3)}`;
        }
    }

    toggleWaveformView() {
        const imageContainer = document.getElementById('waveform-image-container');
        const canvasContainer = imageContainer?.nextElementSibling;
        const toggleBtn = document.getElementById('toggle-waveform-view');

        if (!imageContainer || !canvasContainer || !toggleBtn) return;

        this.showImage = !this.showImage;

        if (this.showImage) {
            imageContainer.classList.remove('hidden');
            canvasContainer.classList.add('hidden');
            toggleBtn.innerHTML = '<i class="fas fa-chart-line mr-2"></i>Hiển thị Canvas';
        } else {
            imageContainer.classList.add('hidden');
            canvasContainer.classList.remove('hidden');
            toggleBtn.innerHTML = '<i class="fas fa-image mr-2"></i>Hiển thị Ảnh';
            
            // Redraw canvases when switching to canvas view
            setTimeout(() => this.drawWaveforms(), 100);
        }
    }

    downloadWaveformImage() {
        if (!this.currentSessionId) {
            console.error('No session ID available for download');
            return;
        }

        const link = document.createElement('a');
        link.href = `/api/waveform-image/${this.currentSessionId}`;
        link.download = `waveform_comparison_${this.currentSessionId}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    clearWaveforms() {
        // Clear canvases
        if (this.originalCtx) {
            this.originalCtx.fillStyle = '#111827';
            this.originalCtx.fillRect(0, 0, this.originalCanvas.width, this.originalCanvas.height);
        }
        if (this.enhancedCtx) {
            this.enhancedCtx.fillStyle = '#111827';
            this.enhancedCtx.fillRect(0, 0, this.enhancedCanvas.width, this.enhancedCanvas.height);
        }

        // Hide image
        const image = document.getElementById('waveform-comparison-image');
        if (image) {
            image.classList.add('hidden');
            image.src = '';
        }

        // Reset data
        this.waveformData = null;
        this.currentSessionId = null;
    }
}

// Initialize waveform visualizer when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.waveformVisualizer = new WaveformVisualizer();
});
