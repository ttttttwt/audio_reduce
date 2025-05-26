// Audio Processing JavaScript
document.addEventListener('DOMContentLoaded', function() {
  // DOM Elements
  const uploadArea = document.getElementById("upload-area");
  const fileInput = document.getElementById("audio-file");
  const fileInfo = document.getElementById("file-info");
  const fileName = document.getElementById("file-name");
  const fileSize = document.getElementById("file-size");
  const removeFileBtn = document.getElementById("remove-file");
  const processBtn = document.getElementById("process-btn");
  const processingSection = document.getElementById("processing-section");
  const resultsSection = document.getElementById("results-section");
  const progressBar = document.getElementById("progress-bar");
  const progressText = document.getElementById("progress-text");

  // Global Variables
  let selectedFile = null;
  let processingStartTime = null;

  // Event Listeners
  uploadArea.addEventListener("click", () => fileInput.click());

  // Drag and drop functionality
  uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("border-green-500");
  });

  uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("border-green-500");
  });

  uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("border-green-500");
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  });

  // File input change
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileSelect(e.target.files[0]);
    }
  });

  // Remove file button
  removeFileBtn.addEventListener("click", () => {
    selectedFile = null;
    fileInput.value = "";
    uploadArea.classList.remove("hidden");
    fileInfo.classList.add("hidden");
    processBtn.disabled = true;
    resultsSection.classList.add("hidden");
  });

  // Process button
  processBtn.addEventListener("click", processAudio);

  // Download button
  document.getElementById("download-btn").addEventListener("click", downloadAudio);

  // File Selection Handler
  function handleFileSelect(file) {
    if (!file.type.startsWith("audio/")) {
      showNotification("Vui lòng chọn file audio hợp lệ", "error");
      return;
    }

    if (file.size > 50 * 1024 * 1024) {
      showNotification("File quá lớn. Vui lòng chọn file nhỏ hơn 50MB", "error");
      return;
    }

    selectedFile = file;
    displayFileInfo(file);
    uploadArea.classList.add("hidden");
    fileInfo.classList.remove("hidden");
    processBtn.disabled = false;
  }

  // Display File Information
  function displayFileInfo(file) {
    // Truncate long filenames for mobile
    const maxLength = window.innerWidth < 640 ? 25 : 50;
    const displayName = file.name.length > maxLength ? 
      file.name.substring(0, maxLength) + "..." : file.name;

    fileName.textContent = displayName;
    fileName.title = file.name; // Show full name on hover
    fileSize.textContent = formatFileSize(file.size);

    // Get audio duration and format
    const audio = document.createElement("audio");
    audio.src = URL.createObjectURL(file);
    audio.addEventListener("loadedmetadata", () => {
      const duration = formatDuration(audio.duration);
      document.getElementById("file-duration").textContent = duration;
      document.getElementById("file-format").textContent = file.type
        .split("/")[1]
        .toUpperCase();
      URL.revokeObjectURL(audio.src);
    });
  }

  // Process Audio Function
  async function processAudio() {
    if (!selectedFile) return;

    processingStartTime = Date.now();
    processingSection.classList.remove("hidden");
    resultsSection.classList.add("hidden");

    try {
      const formData = new FormData();
      formData.append("audio", selectedFile);

      // Start progress animation
      animateProgress();

      const response = await fetch("/api/upload-audio", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (result.success) {
        showResults(result);
      } else {
        throw new Error(result.error || "Unknown error occurred");
      }
    } catch (error) {
      console.error("Processing error:", error);
      showNotification(`Lỗi xử lý: ${error.message}`, "error");
      processingSection.classList.add("hidden");
    }
  }

  // Progress Animation
  function animateProgress() {
    const steps = [
      { step: 1, text: "Đang tải file lên server...", progress: 20 },
      { step: 2, text: "Phân tích tín hiệu âm thanh...", progress: 40 },
      { step: 3, text: "AI đang xử lý và loại bỏ noise...", progress: 80 },
      { step: 4, text: "Tạo file output...", progress: 100 },
    ];

    let currentStep = 0;

    const interval = setInterval(() => {
      if (currentStep < steps.length) {
        const step = steps[currentStep];

        // Update step visual
        document.getElementById(`step-${step.step}`).classList.add("active");
        document.getElementById("processing-step").textContent = step.text;

        // Animate progress bar
        progressBar.style.width = step.progress + "%";
        progressText.textContent = step.progress + "%";

        currentStep++;
      } else {
        clearInterval(interval);
      }
    }, 1500);
  }

  // Show Results
  function showResults(result) {
    processingSection.classList.add("hidden");
    resultsSection.classList.remove("hidden");

    // Update metrics dashboard
    const metrics = result.metrics;
    const processingTime = (Date.now() - processingStartTime) / 1000;

    document.getElementById("snr-improvement").textContent = 
      `+${metrics.snr_improvement_db.toFixed(1)} dB`;
    document.getElementById("noise-reduction").textContent = 
      `${Math.abs(metrics.energy_reduction_percent).toFixed(1)}%`;
    document.getElementById("processing-time").textContent = 
      `${processingTime.toFixed(1)}s`;

    // Calculate quality score (0-100)
    const qualityScore = Math.min(100, Math.max(0, 50 + metrics.snr_improvement_db * 5));
    document.getElementById("quality-score").textContent = qualityScore.toFixed(0);

    // Set original audio
    const originalUrl = URL.createObjectURL(selectedFile);
    document.getElementById("original-audio").src = originalUrl;

    // Set processed audio URL
    const processedAudioUrl = `/api/download-audio/${result.output_file}`;
    document.getElementById("processed-audio").src = processedAudioUrl;

    // Store output filename for download
    window.processedFilename = result.output_file;

    // Display waveforms if available
    if (result.waveforms && window.waveformVisualizer) {
      window.waveformVisualizer.displayWaveforms(
        result.session_id,
        result.waveforms,
        result.waveform_image_available
      );
    }

    showNotification("Xử lý hoàn thành!", "success");
  }

  // Download Audio
  async function downloadAudio() {
    if (!window.processedFilename) {
      showNotification("Không tìm thấy file để tải xuống", "error");
      return;
    }

    try {
      showNotification("Đang tải xuống...", "info");

      const response = await fetch(`/api/download-audio/${window.processedFilename}`);

      if (!response.ok) {
        throw new Error(`Download failed: ${response.status}`);
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `enhanced_${selectedFile.name}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      window.URL.revokeObjectURL(url);
      showNotification("Tải xuống thành công!", "success");
    } catch (error) {
      console.error("Download error:", error);
      showNotification(`Lỗi khi tải xuống file: ${error.message}`, "error");
    }
  }

  // Reset Interface
  function resetInterface() {
    selectedFile = null;
    fileInput.value = "";
    uploadArea.classList.remove("hidden");
    fileInfo.classList.add("hidden");
    processBtn.disabled = true;
    resultsSection.classList.add("hidden");

    // Clear waveforms
    if (window.waveformVisualizer) {
      window.waveformVisualizer.clearWaveforms();
    }
  }

  // Utility Functions
  function formatFileSize(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  function formatDuration(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  }

  function showNotification(message, type = "info") {
    // Remove existing notifications first
    const existingNotifications = document.querySelectorAll(".notification");
    existingNotifications.forEach((notif) => notif.remove());

    const notification = document.createElement("div");
    notification.className = `notification fixed top-4 right-4 left-4 sm:left-auto sm:w-96 z-50 p-3 md:p-4 rounded-lg shadow-lg transition-all transform translate-y-[-100px] opacity-0`;

    if (type === "success") {
      notification.classList.add("bg-green-500", "text-white");
    } else if (type === "error") {
      notification.classList.add("bg-red-500", "text-white");
    } else {
      notification.classList.add("bg-blue-500", "text-white");
    }

    notification.innerHTML = `
      <div class="flex items-center space-x-2">
        <i class="fas fa-${
          type === "success"
            ? "check"
            : type === "error"
            ? "exclamation-triangle"
            : "info"
        }-circle flex-shrink-0"></i>
        <span class="text-sm md:text-base">${message}</span>
      </div>
    `;

    document.body.appendChild(notification);

    // Animate in
    setTimeout(() => {
      notification.classList.remove("translate-y-[-100px]", "opacity-0");
    }, 100);

    // Remove after 4 seconds
    setTimeout(() => {
      notification.classList.add("translate-y-[-100px]", "opacity-0");
      setTimeout(() => {
        if (notification.parentNode) {
          document.body.removeChild(notification);
        }
      }, 300);
    }, 4000);
  }

  // Responsive Handling
  function handleWindowResize() {
    if (selectedFile && fileName) {
      displayFileInfo(selectedFile);
    }
  }

  // Event Listeners
  window.addEventListener("resize", handleWindowResize);

  // Prevent horizontal scroll on mobile
  document.addEventListener("touchmove", function (e) {
    if (e.touches.length > 1) {
      e.preventDefault();
    }
  }, { passive: false });

  // Auto-cleanup on page unload
  window.addEventListener('beforeunload', function() {
    // Cleanup any object URLs
    const audioElements = document.querySelectorAll('audio');
    audioElements.forEach(audio => {
      if (audio.src && audio.src.startsWith('blob:')) {
        URL.revokeObjectURL(audio.src);
      }
    });
  });
});
