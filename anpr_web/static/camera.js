(() => {
  const video = document.querySelector("#camera-preview");
  if (!video) return;
  const canvas = document.querySelector("#camera-canvas");
  const message = document.querySelector("#camera-message");
  const review = document.querySelector("#camera-review");
  const csrf = document.querySelector('input[name="csrf_token"]').value;
  let stream;
  let timer;
  let facingMode = "environment";

  const stop = () => {
    clearInterval(timer);
    timer = undefined;
    if (stream) stream.getTracks().forEach((track) => track.stop());
    stream = undefined;
    video.srcObject = null;
  };

  const open = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      message.textContent = "Camera access is unsupported. Use image upload or manual registration.";
      return false;
    }
    stop();
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode }, audio: false,
      });
      video.srcObject = stream;
      message.textContent = "Camera ready. Frames are sampled conservatively.";
      return true;
    } catch (_error) {
      message.textContent = "Camera permission was denied or unavailable. Use image upload or manual registration.";
      return false;
    }
  };

  const sendFrame = async (manual = false) => {
    if (!stream || video.videoWidth === 0) return;
    const width = Math.min(video.videoWidth, 960);
    const scale = width / video.videoWidth;
    canvas.width = width;
    canvas.height = Math.round(video.videoHeight * scale);
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.78));
    if (!blob) return;
    const body = new FormData();
    body.append("csrf_token", csrf);
    body.append("frame", blob, "camera-frame.jpg");
    body.append("manual", manual ? "1" : "0");
    const response = await fetch("/admin/vehicles/from-camera/frame", {
      method: "POST",
      body,
      credentials: "same-origin",
    });
    const data = await response.json();
    message.textContent = data.message || "Frame analysed.";
    if (data.status === "review") {
      stop();
      review.hidden = false;
      document.querySelector("#camera-original").src = data.original_image || "";
      const crop = document.querySelector("#camera-crop");
      crop.src = data.cropped_image || "";
      crop.closest("figure").hidden = !data.cropped_image;
      document.querySelector("#camera-plate").value = data.plate || "";
    }
    if (data.status === "timeout") clearInterval(timer);
  };

  document.querySelector("#camera-start").addEventListener("click", async () => {
    if (await open()) {
      await sendFrame();
      timer = setInterval(() => sendFrame(), 1200);
    }
  });
  document.querySelector("#camera-capture").addEventListener("click", async () => {
    if (stream || await open()) await sendFrame(true);
  });
  document.querySelector("#camera-switch").addEventListener("click", async () => {
    facingMode = facingMode === "environment" ? "user" : "environment";
    await open();
  });
  document.querySelector("#camera-cancel").addEventListener("click", () => {
    stop();
    window.location.href = "/admin";
  });
  document.querySelector("#camera-retake").addEventListener("click", async () => {
    review.hidden = true;
    await open();
  });
  window.addEventListener("pagehide", stop);
})();
