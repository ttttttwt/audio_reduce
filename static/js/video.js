// Configuration
const config = {
  iceServers: [
    { urls: "stun:stun.l.google.com:19302" },
    { urls: "stun:stun1.l.google.com:19302" },
  ],
};

// Global variables
let roomId;
let socket;
let localStream;
let screenStream;
let isScreenSharing = false;

// Audio processing variables
let isNoiseReductionEnabled = false;
let audioProcessingActive = false;
let audioProcessor = null;
let processingAnalyser = null;

// Get media states from URL parameters
const urlParams = new URLSearchParams(window.location.search);
let localVideoEnabled = urlParams.get("video") === "true";
let localAudioEnabled = urlParams.get("audio") === "true";

// Get username from URL params if available, otherwise generate
let username =
  urlParams.get("username") || "User-" + Math.floor(Math.random() * 1000);

// Store all peer connections
const peerConnections = {};

// Track participants properly
const participants = new Map(); // Use Map to track participants with their info
let participantCount = 0;
let audioContext = null;
let microphoneAnalyser = null;

// DOM elements
let participantsContainer;
let participantCountDisplay;
let roomParticipantCountDisplay;
let toggleAudioBtn;
let toggleVideoBtn;
let shareScreenBtn;
let toggleChatBtn;
let leaveRoomBtn;
let chatContainer;
let chatMessages;
let chatForm;
let chatInput;
let closeChat;
let fullScreenContainer;
let fullScreenVideo;
let exitFullScreen;
let noiseReductionBtn; // New: noise reduction button
let noiseReductionStatus; // New: status indicator

// Update participant count and grid layout
function updateParticipantCount() {
  participantCount = participants.size;
  participantCountDisplay.textContent = participantCount;
  roomParticipantCountDisplay.textContent = participantCount;

  console.log(`Participant count updated: ${participantCount}`);
  console.log("Current participants:", Array.from(participants.keys()));

  updateGridLayout();
}

// Update grid layout based on number of participants with proper calculations
function updateGridLayout() {
  // Remove all existing grid classes
  participantsContainer.classList.remove(
    "grid-1",
    "grid-2",
    "grid-3",
    "grid-4",
    "grid-5",
    "grid-6",
    "grid-7",
    "grid-8",
    "grid-9",
    "grid-10",
    "grid-11",
    "grid-12",
    "grid-13",
    "grid-14",
    "grid-15",
    "grid-16"
  );

  // Apply appropriate grid class based on participant count
  let gridClass = `grid-${Math.min(participantCount, 16)}`;
  participantsContainer.classList.add(gridClass);

  console.log(
    `Grid layout updated: ${gridClass} for ${participantCount} participants`
  );

  // Force layout recalculation
  participantsContainer.offsetHeight;
}

// Initialize
async function init() {
  try {
    // Initialize DOM elements
    initializeDOMElements();

    // Get room ID from page
    roomId = document.getElementById("room-id").textContent;

    // Get local media stream
    localStream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: true,
    });

    // Apply initial states from URL parameters
    localStream.getVideoTracks().forEach((track) => {
      track.enabled = localVideoEnabled;
    });
    localStream.getAudioTracks().forEach((track) => {
      track.enabled = localAudioEnabled;
    });

    // Create local participant view
    createLocalParticipant();

    // Add local participant to tracking
    participants.set("local", {
      id: "local",
      username: username,
      isLocal: true,
    });

    updateParticipantCount();

    // Connect to Socket.IO server
    connectSocket();

    // Set up UI event listeners
    setupEventListeners();

    // Initialize microphone visualization
    if (localAudioEnabled) {
      initializeMicrophoneVisualization();
    }

    // Update UI buttons to reflect initial states
    updateToggleButtons();
  } catch (error) {
    console.error("Error initializing:", error);
    alert(
      "Could not access camera/microphone. Please check your permissions."
    );
  }
}

// Initialize DOM elements
function initializeDOMElements() {
  participantsContainer = document.getElementById("participants-container");
  participantCountDisplay = document.getElementById("participant-count");
  roomParticipantCountDisplay = document.getElementById("room-participant-count");
  toggleAudioBtn = document.getElementById("toggle-audio");
  toggleVideoBtn = document.getElementById("toggle-video");
  shareScreenBtn = document.getElementById("share-screen");
  toggleChatBtn = document.getElementById("toggle-chat");
  leaveRoomBtn = document.getElementById("leave-room");
  chatContainer = document.getElementById("chat-container");
  chatMessages = document.getElementById("chat-messages");
  chatForm = document.getElementById("chat-form");
  chatInput = document.getElementById("chat-input");
  closeChat = document.getElementById("close-chat");
  fullScreenContainer = document.getElementById("full-screen-container");
  fullScreenVideo = document.getElementById("full-screen-video");
  exitFullScreen = document.getElementById("exit-full-screen");
  noiseReductionBtn = document.getElementById("noise-reduction-btn");
  noiseReductionStatus = document.getElementById("noise-reduction-status");
}

// Update toggle buttons to reflect current states
function updateToggleButtons() {
  // Update audio button
  if (!localAudioEnabled) {
    toggleAudioBtn.innerHTML =
      '<i class="fas fa-microphone-slash"></i><span class="mic-animation"></span>';
    toggleAudioBtn.classList.add("active");
    document
      .getElementById("local-mic-icon")
      .classList.add("mic-disabled");
  } else {
    toggleAudioBtn.innerHTML =
      '<i class="fas fa-microphone"></i><span class="mic-animation"></span>';
    toggleAudioBtn.classList.remove("active");
    document
      .getElementById("local-mic-icon")
      .classList.remove("mic-disabled");
  }

  // Update video button
  if (!localVideoEnabled) {
    toggleVideoBtn.innerHTML = '<i class="fas fa-video-slash"></i>';
    toggleVideoBtn.classList.add("active");
    document
      .getElementById("local-video-icon")
      .classList.add("camera-disabled");
  } else {
    toggleVideoBtn.innerHTML = '<i class="fas fa-video"></i>';
    toggleVideoBtn.classList.remove("active");
    document
      .getElementById("local-video-icon")
      .classList.remove("camera-disabled");
  }
}

// Initialize microphone visualization
function initializeMicrophoneVisualization() {
  if (!localAudioEnabled || !localStream) return;

  try {
    audioContext = new (window.AudioContext ||
      window.webkitAudioContext)();
    microphoneAnalyser = audioContext.createAnalyser();
    const microphone = audioContext.createMediaStreamSource(localStream);
    microphone.connect(microphoneAnalyser);

    microphoneAnalyser.fftSize = 256;
    const bufferLength = microphoneAnalyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function animate() {
      if (!localAudioEnabled) {
        toggleAudioBtn.classList.remove("speaking");
        return;
      }

      requestAnimationFrame(animate);
      microphoneAnalyser.getByteFrequencyData(dataArray);

      // Calculate average frequency
      let sum = 0;
      for (let i = 0; i < bufferLength; i++) {
        sum += dataArray[i];
      }
      const average = sum / bufferLength;

      // Add visual feedback when speaking
      if (average > 35) {
        toggleAudioBtn.classList.add("speaking");
      } else {
        toggleAudioBtn.classList.remove("speaking");
      }

      // Update animation intensity
      const scale = 1 + (average / 128) * 0.3;
      const micAnimation = toggleAudioBtn.querySelector(".mic-animation");
      if (micAnimation) {
        micAnimation.style.transform = `translate(-50%, -50%) scale(${scale})`;
      }
    }

    animate();
  } catch (error) {
    console.error("Error initializing microphone visualization:", error);
  }
}

// Create local participant view
function createLocalParticipant() {
  const participantDiv = document.createElement("div");
  participantDiv.id = "local-participant";
  participantDiv.className = "participant-item";

  const video = document.createElement("video");
  video.id = "local-video";
  video.className = "participant-video";
  video.autoplay = true;
  video.playsInline = true;
  video.muted = true; // Mute local video to prevent feedback

  const infoBar = document.createElement("div");
  infoBar.className = "participant-info";
  infoBar.innerHTML = `
    <div>
      <span class="font-semibold">You (${username})</span>
    </div>
    <div class="flex space-x-2">
      <span id="local-mic-icon" class="relative">
        <i class="fas fa-microphone text-sm"></i>
      </span>
      <span id="local-video-icon" class="relative">
        <i class="fas fa-video text-sm"></i>
      </span>
      <span class="relative noise-reduction-icon hidden" title="AI Noise Reduction">
        <i class="fas fa-magic text-sm text-purple-400"></i>
      </span>
    </div>
  `;

  participantDiv.appendChild(video);
  participantDiv.appendChild(infoBar);
  participantsContainer.appendChild(participantDiv);

  // Add local stream to video element
  video.srcObject = localStream;
}

// Connect to Socket.IO server
function connectSocket() {
  socket = io.connect(window.location.origin);

  // Socket event handlers
  socket.on("connect", () => {
    console.log("Connected to server with ID:", socket.id);
    // Join room when connected
    socket.emit("join_room", {
      room_id: roomId,
      username: username,
      video_enabled: localVideoEnabled,
      audio_enabled: localAudioEnabled,
    });
  });

  // When user joins room
  socket.on("user_joined", async (data) => {
    console.log("User joined event received:", data);

    // Clear and rebuild participant list to ensure accuracy
    // First, keep track of existing remote participants
    const existingRemoteParticipants = new Set();

    // Process all participants from server
    if (data.participants && Array.isArray(data.participants)) {
      data.participants.forEach((participant) => {
        if (participant.user_id !== socket.id) {
          existingRemoteParticipants.add(participant.user_id);

          // Add to our tracking if not already present
          if (!participants.has(participant.user_id)) {
            participants.set(participant.user_id, {
              id: participant.user_id,
              username: participant.username,
              isLocal: false,
            });

            // Create UI element if it doesn't exist
            if (
              !document.getElementById(
                `participant-${participant.user_id}`
              )
            ) {
              createRemoteParticipant(
                participant.user_id,
                participant.username
              );
            }
          }
        }
      });
    }

    // Remove participants that are no longer in the room
    const toRemove = [];
    participants.forEach((participant, id) => {
      if (!participant.isLocal && !existingRemoteParticipants.has(id)) {
        toRemove.push(id);
      }
    });

    toRemove.forEach((id) => {
      participants.delete(id);
      const element = document.getElementById(`participant-${id}`);
      if (element) {
        element.remove();
      }
      if (peerConnections[id]) {
        peerConnections[id].close();
        delete peerConnections[id];
      }
    });

    updateParticipantCount();

    // For new users only, create peer connection
    if (data.user_id !== socket.id && !peerConnections[data.user_id]) {
      console.log("Creating peer connection for new user:", data.user_id);
      const peerConnection = createPeerConnection(data.user_id);

      localStream.getTracks().forEach((track) => {
        peerConnection.addTrack(track, localStream);
      });

      try {
        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);

        socket.emit("offer", {
          target: data.user_id,
          caller: socket.id,
          sdp: peerConnection.localDescription,
        });
      } catch (error) {
        console.error("Error creating offer:", error);
      }
    }
  });

  // When user leaves room
  socket.on("user_left", (data) => {
    console.log("User left:", data);

    // Remove participant from tracking
    if (participants.has(data.user_id)) {
      participants.delete(data.user_id);
    }

    // Remove participant from grid
    const participantElement = document.getElementById(
      `participant-${data.user_id}`
    );
    if (participantElement) {
      participantElement.remove();
    }

    // Close and remove peer connection
    if (peerConnections[data.user_id]) {
      peerConnections[data.user_id].close();
      delete peerConnections[data.user_id];
    }

    updateParticipantCount();
  });

  // WebRTC Signaling - Handle offers
  socket.on("offer", async (data) => {
    if (data.caller !== socket.id) {
      console.log("Received offer:", data);

      // Create peer connection if it doesn't exist
      const peerConnection =
        peerConnections[data.caller] || createPeerConnection(data.caller);

      try {
        // Set remote description based on offer
        await peerConnection.setRemoteDescription(
          new RTCSessionDescription(data.sdp)
        );

        // Add local tracks to connection
        localStream.getTracks().forEach((track) => {
          peerConnection.addTrack(track, localStream);
        });

        // Create and send answer
        const answer = await peerConnection.createAnswer();
        await peerConnection.setLocalDescription(answer);

        socket.emit("answer", {
          target: data.caller,
          caller: socket.id,
          sdp: peerConnection.localDescription,
        });
      } catch (error) {
        console.error("Error handling offer:", error);
      }
    }
  });

  // Handle answers to our offers
  socket.on("answer", async (data) => {
    console.log("Received answer:", data);

    if (peerConnections[data.caller]) {
      try {
        await peerConnections[data.caller].setRemoteDescription(
          new RTCSessionDescription(data.sdp)
        );
      } catch (error) {
        console.error("Error setting remote description:", error);
      }
    }
  });

  // Handle ICE candidates
  socket.on("ice_candidate", async (data) => {
    console.log("Received ICE candidate:", data);

    if (peerConnections[data.caller]) {
      try {
        await peerConnections[data.caller].addIceCandidate(
          new RTCIceCandidate(data.candidate)
        );
      } catch (error) {
        console.error("Error adding ICE candidate:", error);
      }
    }
  });

  // Handle chat messages
  socket.on("new_message", (message) => {
    addMessageToChat(message);
  });

  socket.on("chat_history", (data) => {
    data.messages.forEach((message) => {
      addMessageToChat(message);
    });
  });

  // Handle media controls
  socket.on("user_video_toggle", (data) => {
    const videoIcon = document.querySelector(
      `#participant-${data.user_id} .video-icon`
    );
    if (videoIcon) {
      if (!data.enabled) {
        videoIcon.classList.add("camera-disabled");
      } else {
        videoIcon.classList.remove("camera-disabled");
      }
    }
  });

  socket.on("user_audio_toggle", (data) => {
    const micIcon = document.querySelector(
      `#participant-${data.user_id} .mic-icon`
    );
    if (micIcon) {
      if (!data.enabled) {
        micIcon.classList.add("mic-disabled");
      } else {
        micIcon.classList.remove("mic-disabled");
      }
    }
  });

  // Handle screen sharing events
  socket.on("user_screen_sharing", (data) => {
    const participantElement = document.getElementById(
      `participant-${data.user_id}`
    );
    if (participantElement) {
      if (data.sharing) {
        participantElement.classList.add("screen-sharing");
      } else {
        participantElement.classList.remove("screen-sharing");
      }
    }
  });

  // New: AI Noise Reduction Events
  socket.on("video_noise_reduction_started", (data) => {
    console.log("✅ Video noise reduction started:", data);
    audioProcessingActive = true;
    isNoiseReductionEnabled = true;
    
    updateNoiseReductionUI(true);
    showVideoNotification("AI Noise Reduction đã được bật", "success");
    
    // Start processing audio if microphone is active
    if (isMicActive && localStream) {
        startAudioProcessing();
    }
  });

  socket.on("video_noise_reduction_stopped", (data) => {
    console.log("🛑 Video noise reduction stopped:", data);
    audioProcessingActive = false;
    isNoiseReductionEnabled = false;
    
    updateNoiseReductionUI(false);
    showVideoNotification("AI Noise Reduction đã được tắt", "info");
    
    stopAudioProcessing();
  });

  socket.on("video_audio_processed", (data) => {
    // Handle processed audio and update any UI metrics
    if (data.metrics) {
        updateAudioMetrics(data.metrics);
    }
  });

  socket.on("noise_reduction_error", (data) => {
    console.error("❌ Noise reduction error:", data.error);
    showVideoNotification("Lỗi AI Noise Reduction: " + data.error, "error");
    
    // Reset state on error
    audioProcessingActive = false;
    isNoiseReductionEnabled = false;
    updateNoiseReductionUI(false);
  });

  socket.on("user_noise_reduction_toggle", (data) => {
    // Update UI to show which users have noise reduction enabled
    const participantElement = document.getElementById(`participant-${data.user_id}`);
    if (participantElement) {
        const noiseIcon = participantElement.querySelector('.noise-reduction-icon');
        if (noiseIcon) {
            if (data.enabled) {
                noiseIcon.classList.remove('hidden');
                noiseIcon.title = `${data.username} đang sử dụng AI Noise Reduction`;
            } else {
                noiseIcon.classList.add('hidden');
            }
        }
    }
  });

  socket.on("video_audio_status", (data) => {
    console.log("Audio processor status:", data);
    
    if (data.processor_available) {
        noiseReductionBtn.disabled = false;
        noiseReductionStatus.textContent = `AI Ready (${data.device})`;
        noiseReductionStatus.className = "text-xs text-green-400";
    } else {
        noiseReductionBtn.disabled = true;
        noiseReductionStatus.textContent = "AI Không khả dụng";
        noiseReductionStatus.className = "text-xs text-red-400";
    }
  });
}

// Create a peer connection for a specific user
function createPeerConnection(userId) {
  if (peerConnections[userId]) {
    peerConnections[userId].close();
  }

  const peerConnection = new RTCPeerConnection(config);
  peerConnections[userId] = peerConnection;

  // Handle ICE candidates
  peerConnection.onicecandidate = (event) => {
    if (event.candidate) {
      socket.emit("ice_candidate", {
        target: userId,
        caller: socket.id,
        candidate: event.candidate,
      });
    }
  };

  // Handle connection state changes
  peerConnection.onconnectionstatechange = () => {
    console.log(
      `Connection state change: ${peerConnection.connectionState}`
    );
  };

  // Handle incoming tracks
  peerConnection.ontrack = (event) => {
    console.log("Got remote track:", event);

    const remoteVideo = document.querySelector(
      `#participant-${userId} video`
    );
    if (remoteVideo) {
      remoteVideo.srcObject = event.streams[0];
    }
  };

  return peerConnection;
}

// Create a new remote participant in the grid
function createRemoteParticipant(userId, username) {
    console.log(`Creating remote participant: ${userId} - ${username}`);

    // Check if participant already exists
    if (document.getElementById(`participant-${userId}`)) {
        console.log(`Participant ${userId} already exists, skipping creation`);
        return;
    }

    const participantDiv = document.createElement("div");
    participantDiv.id = `participant-${userId}`;
    participantDiv.className = "participant-item";

    const video = document.createElement("video");
    video.className = "participant-video";
    video.autoplay = true;
    video.playsInline = true;

    const infoBar = document.createElement("div");
    infoBar.className = "participant-info";
    infoBar.innerHTML = `
        <div>
            <span class="font-semibold">${username}</span>
        </div>
        <div class="flex space-x-2">
            <span class="relative mic-icon">
                <i class="fas fa-microphone text-sm"></i>
            </span>
            <span class="relative video-icon">
                <i class="fas fa-video text-sm"></i>
            </span>
            <span class="relative noise-reduction-icon hidden" title="AI Noise Reduction">
                <i class="fas fa-magic text-sm text-purple-400"></i>
            </span>
        </div>
    `;

    participantDiv.appendChild(video);
    participantDiv.appendChild(infoBar);

    // Add click event to view in full screen
    video.addEventListener("click", () => {
        if (video.srcObject) {
            fullScreenVideo.srcObject = video.srcObject;
            fullScreenContainer.classList.remove("hidden");
        }
    });

    participantsContainer.appendChild(participantDiv);

    console.log(`Remote participant ${userId} created successfully`);
}

// Function to stop screen sharing
function stopScreenSharing(originalVideoTrack) {
  if (!isScreenSharing) return;

  // Replace screen share with camera in all peer connections
  const senders = Object.values(peerConnections);

  senders.forEach((sender) => {
    const videoSender = sender
      .getSenders()
      .find((s) => s.track && s.track.kind === "video");
    if (videoSender) {
      videoSender.replaceTrack(originalVideoTrack);
    }
  });

  // Update local video
  const localVideo = document.getElementById("local-video");
  const localStreamTracks = localVideo.srcObject.getTracks();
  localStreamTracks.forEach((track) => {
    if (track.kind === "video") {
      track.stop();
      localVideo.srcObject.removeTrack(track);
    }
  });
  localVideo.srcObject.addTrack(originalVideoTrack);

  // Clean up screen stream
  if (screenStream) {
    screenStream.getTracks().forEach((track) => track.stop());
    screenStream = null;
  }

  // Update UI
  shareScreenBtn.innerHTML = '<i class="fas fa-desktop"></i>';
  shareScreenBtn.classList.remove("active");
  isScreenSharing = false;

  // Notify server
  socket.emit("stop_screen_sharing");
}

// Set up UI event listeners
function setupEventListeners() {
  // Toggle audio
  toggleAudioBtn.addEventListener("click", () => {
    localAudioEnabled = !localAudioEnabled;

    // Update UI
    if (!localAudioEnabled) {
      toggleAudioBtn.innerHTML =
        '<i class="fas fa-microphone-slash"></i><span class="mic-animation"></span>';
      toggleAudioBtn.classList.add("active");
      document
        .getElementById("local-mic-icon")
        .classList.add("mic-disabled");

      // Stop microphone visualization
      if (audioContext) {
        audioContext.suspend();
      }
    } else {
      toggleAudioBtn.innerHTML =
        '<i class="fas fa-microphone"></i><span class="mic-animation"></span>';
      toggleAudioBtn.classList.remove("active");
      document
        .getElementById("local-mic-icon")
        .classList.remove("mic-disabled");

      // Resume microphone visualization
      if (audioContext) {
        audioContext.resume();
        initializeMicrophoneVisualization();
      } else {
        initializeMicrophoneVisualization();
      }
    }

    // Mute/unmute local audio tracks
    localStream.getAudioTracks().forEach((track) => {
      track.enabled = localAudioEnabled;
    });

    // Notify server
    socket.emit("toggle_audio", { enabled: localAudioEnabled });
  });

  // Toggle video
  toggleVideoBtn.addEventListener("click", () => {
    localVideoEnabled = !localVideoEnabled;

    // Update UI
    if (!localVideoEnabled) {
      toggleVideoBtn.innerHTML = '<i class="fas fa-video-slash"></i>';
      toggleVideoBtn.classList.add("active");
      document
        .getElementById("local-video-icon")
        .classList.add("camera-disabled");
    } else {
      toggleVideoBtn.innerHTML = '<i class="fas fa-video"></i>';
      toggleVideoBtn.classList.remove("active");
      document
        .getElementById("local-video-icon")
        .classList.remove("camera-disabled");
    }

    // Disable/enable local video tracks
    localStream.getVideoTracks().forEach((track) => {
      track.enabled = localVideoEnabled;
    });

    // Notify server
    socket.emit("toggle_video", { enabled: localVideoEnabled });
  });

  // Screen sharing
  shareScreenBtn.addEventListener("click", async () => {
    if (!isScreenSharing) {
      try {
        // Get screen capture stream
        screenStream = await navigator.mediaDevices.getDisplayMedia({
          video: { cursor: "always" },
          audio: false,
        });

        // Replace video track in all peer connections
        const videoTrack = screenStream.getVideoTracks()[0];

        // Replace track in local video
        const localVideo = document.getElementById("local-video");
        const localVideoStream = localVideo.srcObject;
        const senders = Object.values(peerConnections);

        // Store original video track to restore later
        const originalVideoTrack = localStream.getVideoTracks()[0];

        // Replace tracks in all peer connections
        senders.forEach((sender) => {
          const videoSender = sender
            .getSenders()
            .find((s) => s.track && s.track.kind === "video");
          if (videoSender) {
            videoSender.replaceTrack(videoTrack);
          }
        });

        // Update local stream
        const localStreamTracks = localVideo.srcObject.getTracks();
        localStreamTracks.forEach((track) => {
          if (track.kind === "video") {
            localVideo.srcObject.removeTrack(track);
          }
        });
        localVideo.srcObject.addTrack(videoTrack);

        // Update UI
        shareScreenBtn.innerHTML =
          '<i class="fas fa-desktop"></i> <i class="fas fa-stop text-xs absolute bottom-0 right-0"></i>';
        shareScreenBtn.classList.add("active");
        isScreenSharing = true;

        // Notify server
        socket.emit("start_screen_sharing");

        // Handle the end of screen sharing
        videoTrack.onended = () => {
          stopScreenSharing(originalVideoTrack);
        };
      } catch (error) {
        console.error("Error sharing screen:", error);
      }
    } else {
      // Get original video track
      const originalVideoTrack = localStream.getVideoTracks()[0];
      stopScreenSharing(originalVideoTrack);
    }
  });

  // Toggle chat
  toggleChatBtn.addEventListener("click", () => {
    chatContainer.classList.toggle("open");
  });

  // Close chat button
  closeChat.addEventListener("click", () => {
    chatContainer.classList.remove("open");
  });

  // Submit chat message
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const message = chatInput.value.trim();
    if (message) {
      socket.emit("send_message", {
        message: message,
        timestamp: new Date().toISOString(),
      });
      chatInput.value = "";
    }
  });

  // Exit full screen
  exitFullScreen.addEventListener("click", () => {
    fullScreenContainer.classList.add("hidden");
    fullScreenVideo.srcObject = null;
  });

  // Leave room
  leaveRoomBtn.addEventListener("click", () => {
    // Clean up participants tracking
    participants.clear();
    updateParticipantCount();

    socket.emit("leave_room");

    // Clean up streams
    if (localStream) {
      localStream.getTracks().forEach((track) => track.stop());
    }

    if (screenStream) {
      screenStream.getTracks().forEach((track) => track.stop());
    }

    // Clean up audio context
    if (audioContext) {
      audioContext.close();
    }

    // Close connections
    Object.values(peerConnections).forEach((connection) => {
      connection.close();
    });

    // Redirect to home
    window.location.href = "/";
  });

  // New: Noise Reduction Toggle
  noiseReductionBtn.addEventListener("click", () => {
    toggleNoiseReduction();
  });
}

// New: Toggle noise reduction functionality
async function toggleNoiseReduction() {
    if (!audioProcessingActive) {
        try {
            const settings = {
                chunk_size: 2048,
                sample_rate: 16000
            };

            socket.emit("start_video_noise_reduction", { settings });
            noiseReductionBtn.innerHTML = '<i class="fas fa-magic"></i><span class="loading-spinner"></span>';
            noiseReductionBtn.disabled = true;
            noiseReductionStatus.textContent = "Đang khởi động...";
            noiseReductionStatus.className = "text-xs text-yellow-400";

        } catch (error) {
            console.error("Error starting noise reduction:", error);
            showVideoNotification("Lỗi khởi động AI Noise Reduction", "error");
        }
    } else {
        try {
            socket.emit("stop_video_noise_reduction");
            updateNoiseReductionUI(false);
        } catch (error) {
            console.error("Error stopping noise reduction:", error);
        }
    }
}

// New: Update noise reduction UI
function updateNoiseReductionUI(enabled) {
    if (enabled) {
        noiseReductionBtn.innerHTML = '<i class="fas fa-magic"></i>';
        noiseReductionBtn.classList.add("active");
        noiseReductionBtn.disabled = false;
        noiseReductionStatus.textContent = "AI Đang hoạt động";
        noiseReductionStatus.className = "text-xs text-purple-400";
        
        // Show noise reduction icon for local user
        const localNoiseIcon = document.querySelector('#local-participant .noise-reduction-icon');
        if (localNoiseIcon) {
            localNoiseIcon.classList.remove('hidden');
        }
    } else {
        noiseReductionBtn.innerHTML = '<i class="fas fa-magic"></i>';
        noiseReductionBtn.classList.remove("active");
        noiseReductionBtn.disabled = false;
        noiseReductionStatus.textContent = "AI Sẵn sàng";
        noiseReductionStatus.className = "text-xs text-gray-400";
        
        // Hide noise reduction icon for local user
        const localNoiseIcon = document.querySelector('#local-participant .noise-reduction-icon');
        if (localNoiseIcon) {
            localNoiseIcon.classList.add('hidden');
        }
    }
}

// New: Start audio processing for noise reduction
function startAudioProcessing() {
    if (!isNoiseReductionEnabled || !localStream) return;

    try {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        processingAnalyser = audioContext.createAnalyser();
        const microphone = audioContext.createMediaStreamSource(localStream);
        microphone.connect(processingAnalyser);

        const chunkSize = 2048;
        const processor = audioContext.createScriptProcessor(chunkSize, 1, 1);

        processor.onaudioprocess = (event) => {
            if (isNoiseReductionEnabled && audioProcessingActive) {
                const inputBuffer = event.inputBuffer.getChannelData(0);
                processAudioForNoiseReduction(inputBuffer);
            }
        };

        microphone.connect(processor);
        processor.connect(audioContext.destination);

        audioProcessor = {
            audioContext,
            processor,
            microphone
        };

        console.log("✅ Audio processing started for noise reduction");

    } catch (error) {
        console.error("Error starting audio processing:", error);
    }
}

// New: Stop audio processing
function stopAudioProcessing() {
    if (audioProcessor) {
        try {
            audioProcessor.processor.disconnect();
            audioProcessor.audioContext.close();
            audioProcessor = null;
            console.log("🛑 Audio processing stopped");
        } catch (error) {
            console.error("Error stopping audio processing:", error);
        }
    }
}

// New: Process audio chunk for noise reduction
function processAudioForNoiseReduction(audioData) {
    try {
        // Convert Float32Array to base64
        const bytes = new Uint8Array(audioData.buffer);
        const base64Audio = btoa(String.fromCharCode.apply(null, bytes));

        socket.emit("process_video_audio_chunk", {
            audio_data: base64Audio,
            sample_rate: 16000
        });

    } catch (error) {
        console.error("Error processing audio chunk:", error);
    }
}

// New: Update audio metrics display
function updateAudioMetrics(metrics) {
    // You can add UI elements to show these metrics if needed
    console.log("Audio metrics:", metrics);
}

// New: Check audio processor status on join
function checkAudioProcessorStatus() {
    socket.emit("get_video_audio_status");
}

// New: Show notifications specific to video calls
function showVideoNotification(message, type = "info") {
    const colors = {
        success: "bg-green-600",
        error: "bg-red-600",
        warning: "bg-yellow-600",
        info: "bg-blue-600",
    };

    const notification = document.createElement("div");
    notification.className = `fixed top-4 right-4 ${colors[type]} text-white px-4 py-2 rounded-lg shadow-lg z-50 transition-all duration-300`;
    notification.innerHTML = `<i class="fas fa-magic mr-2"></i>${message}`;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.opacity = "0";
        notification.style.transform = "translateX(100%)";
        setTimeout(() => {
            if (document.body.contains(notification)) {
                document.body.removeChild(notification);
            }
        }, 300);
    }, 3000);
}

// Update initialization to include audio processing check
async function init() {
    try {
        // Initialize DOM elements
        initializeDOMElements();

        // Get room ID from page
        roomId = document.getElementById("room-id").textContent;

        // Get local media stream
        localStream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: true,
        });

        // Apply initial states from URL parameters
        localStream.getVideoTracks().forEach((track) => {
          track.enabled = localVideoEnabled;
        });
        localStream.getAudioTracks().forEach((track) => {
          track.enabled = localAudioEnabled;
        });

        // Create local participant view
        createLocalParticipant();

        // Add local participant to tracking
        participants.set("local", {
          id: "local",
          username: username,
          isLocal: true,
        });

        updateParticipantCount();

        // Connect to Socket.IO server
        connectSocket();

        // Set up UI event listeners
        setupEventListeners();

        // Initialize microphone visualization
        if (localAudioEnabled) {
          initializeMicrophoneVisualization();
        }

        // Update UI buttons to reflect initial states
        updateToggleButtons();

        // Check audio processor status after connecting
        setTimeout(() => {
            checkAudioProcessorStatus();
        }, 1000);

    } catch (error) {
        console.error("Error initializing:", error);
        alert("Could not access camera/microphone. Please check your permissions.");
    }
}

// Update cleanup on page unload
window.addEventListener("beforeunload", () => {
    if (audioProcessingActive) {
        socket.emit("stop_video_noise_reduction");
    }
    
    stopAudioProcessing();
    
    // Clean up participants tracking
    participants.clear();
    updateParticipantCount();

    // Leave room
    socket.emit("leave_room");

    // Clean up streams
    if (localStream) {
      localStream.getTracks().forEach((track) => track.stop());
    }

    if (screenStream) {
      screenStream.getTracks().forEach((track) => track.stop());
    }

    // Clean up audio context
    if (audioContext) {
      audioContext.close();
    }

    // Close connections
    Object.values(peerConnections).forEach((connection) => {
      connection.close();
    });
});

// Add a message to the chat
function addMessageToChat(message) {
  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${
    message.user_id === socket.id ? "own" : ""
  }`;

  const timestamp = new Date(message.timestamp).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });

  messageDiv.innerHTML = `
    <div class="text-xs text-gray-400">${message.username} • ${timestamp}</div>
    <div class="mt-1">${message.message}</div>
  `;

  chatMessages.appendChild(messageDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Initialize when DOM is loaded
document.addEventListener("DOMContentLoaded", init);

// Handle window resize
window.addEventListener("resize", () => {
  // Trigger grid layout update on resize
  setTimeout(updateGridLayout, 100);
});

// Debug function to check participant state
window.debugParticipants = function () {
  console.log("Current participants:", participants);
  console.log("Participant count:", participantCount);
  console.log(
    "DOM elements:",
    document.querySelectorAll(".participant-item").length
  );
};
