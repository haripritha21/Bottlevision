import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_BASE = "https://bottlevisio.onrender.com";
const STATUS_TIMEOUT_MS = 5000;
const ANALYSIS_TIMEOUT_MS = 120000;
const ANALYSIS_POLL_INTERVAL_MS = 700;

async function fetchWithTimeout(url, options = {}, timeoutMs = STATUS_TIMEOUT_MS) {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    window.clearTimeout(timeoutId);
  }
}

function formatSeconds(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "--";
  return `${value.toFixed(2)}s`;
}

function getCameraStatusLabel(statusData) {
  if (statusData?.camera_active) return "Active";
  if ((statusData?.message || "").toLowerCase().includes("no camera")) return "Unavailable";
  return "Waiting";
}

function isAnalysisRunning(statusData) {
  return ["queued", "preparing", "processing"].includes(statusData?.state || "");
}

function App() {
  const [count, setCount] = useState(0);
  const [cameraStatus, setCameraStatus] = useState("Checking...");
  const [statusMessage, setStatusMessage] = useState("Starting backend check...");
  const [backendReachable, setBackendReachable] = useState(false);
  const [streamError, setStreamError] = useState("");

  // dropdown
  const [videos, setVideos] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState("");

  const [analysisResult, setAnalysisResult] = useState(null);
  const [analysisProgress, setAnalysisProgress] = useState(null);
  const [analysisError, setAnalysisError] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisJobId, setAnalysisJobId] = useState("");
  const [analysisStreamUrl, setAnalysisStreamUrl] = useState("");
  const [analysisStreamError, setAnalysisStreamError] = useState("");

  const detectedVideoRef = useRef(null);

  // ── live status poll ──────────────────────────────────────────
  useEffect(() => {
    let disposed = false;

    const fetchData = async () => {
      try {
        const res = await fetchWithTimeout(`${API_BASE}/status`);
        const data = await res.json();
        if (disposed) return;
        setBackendReachable(true);
        setCameraStatus(getCameraStatusLabel(data));
        setStatusMessage(data.message || "Backend connected");
        if (typeof data.count !== "undefined") setCount(data.count);
      } catch {
        if (disposed) return;
        setBackendReachable(false);
        setCameraStatus("Disconnected");
        setStatusMessage("Backend is not reachable");
      }
    };

    fetchData();
    const id = setInterval(fetchData, 1000);
    return () => { disposed = true; clearInterval(id); };
  }, []);

  // ── load assets videos for dropdown ──────────────────────────
  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch(`${API_BASE}/videos`);
        const data = await res.json();
        setVideos(data.videos || []);
      } catch {
        setVideos([]);
      }
    };
    load();
  }, []);

  // ── auto-play detected video ──────────────────────────────────
  useEffect(() => {
    const player = detectedVideoRef.current;
    const url = analysisResult?.annotated_video_url;
    if (!player || !url) return;
    const tryPlay = () => player.play().catch(() => {});
    player.currentTime = 0;
    tryPlay();
    player.addEventListener("loadeddata", tryPlay);
    return () => player.removeEventListener("loadeddata", tryPlay);
  }, [analysisResult?.annotated_video_url]);

  // ── analysis job poll ─────────────────────────────────────────
  useEffect(() => {
    if (!analysisJobId) return;
    let disposed = false;
    let intervalId = 0;

    const poll = async () => {
      try {
        const res = await fetchWithTimeout(`${API_BASE}/analysis_jobs/${analysisJobId}`);
        const data = await res.json();
        if (disposed) return;
        if (!res.ok) throw new Error(data.detail || "Status error");
        setAnalysisProgress(data);
        if (data.state === "completed") {
          setAnalysisResult(data);
          setAnalysisError("");
          setIsAnalyzing(false);
          clearInterval(intervalId);
        } else if (data.state === "failed") {
          setAnalysisResult(null);
          setAnalysisError(data.error || data.message || "Analysis failed.");
          setIsAnalyzing(false);
          clearInterval(intervalId);
        }
      } catch (err) {
        if (disposed) return;
        setAnalysisError(err.message || "Unable to track progress.");
        setIsAnalyzing(false);
        clearInterval(intervalId);
      }
    };

    poll();
    intervalId = setInterval(poll, ANALYSIS_POLL_INTERVAL_MS);
    return () => { disposed = true; clearInterval(intervalId); };
  }, [analysisJobId]);

  // ── handle dropdown change ────────────────────────────────────
  const handleVideoChange = (e) => {
    setSelectedVideo(e.target.value);
    setAnalysisResult(null);
    setAnalysisProgress(null);
    setAnalysisError("");
    setAnalysisJobId("");
    setAnalysisStreamUrl("");
    setAnalysisStreamError("");
    setIsAnalyzing(false);
  };
  const handleReset = async () => {
  try {
    await fetch(`${API_BASE}/reset_analysis`, { method: "POST" });
    setAnalysisError("");
    setAnalysisResult(null);
    setAnalysisProgress(null);
    setAnalysisJobId("");
    setAnalysisStreamUrl("");
    setAnalysisStreamError("");
    setIsAnalyzing(false);
  } catch {
    setAnalysisError("Reset failed. Try again.");
  }
};

  // ── submit: analyze selected assets video ────────────────────
  const handleAnalyze = async (e) => {
    e.preventDefault();

    if (!selectedVideo) {
      setAnalysisError("Please select a video from the list.");
      return;
    }

    setIsAnalyzing(true);
    setAnalysisError("");
    setAnalysisResult(null);
    setAnalysisProgress(null);
    setAnalysisJobId("");
    setAnalysisStreamUrl("");
    setAnalysisStreamError("");

    try {
      const res = await fetchWithTimeout(
        `${API_BASE}/analyze_assets_video`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ video: selectedVideo }),
        },
        ANALYSIS_TIMEOUT_MS,
      );

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Video analysis failed.");

      setAnalysisJobId(data.job_id || "");
      setAnalysisStreamUrl(data.stream_url ? `${API_BASE}${data.stream_url}` : "");
      setAnalysisProgress({ state: "queued", message: "Upload received. Preparing live detection..." });
    } catch (err) {
      setAnalysisError(err.message || "Unable to analyze the selected video.");
      setIsAnalyzing(false);
    }
  };

  const liveUploadStatus = analysisResult || analysisProgress;
  const showLiveStream = Boolean(analysisStreamUrl) && isAnalysisRunning(analysisProgress);

  return (
    <div className="app-shell">
      <div className="navbar">Bottle Counter Dashboard</div>

      {/* ── live camera ── */}
      <div className="container">
        <div className="video-box">
          <h2>Live Camera Feed</h2>
          {backendReachable && !streamError ? (
            <img
              src={`${API_BASE}/video`}
              alt="live-video"
              className="camera-feed"
              onLoad={() => setStreamError("")}
              onError={() => setStreamError("Live stream could not be loaded from the backend.")}
            />
          ) : (
            <div className="camera-placeholder">
              {backendReachable
                ? streamError || "Live stream is unavailable right now."
                : "Backend is offline. Start FastAPI on port 8000 to load the live feed."}
            </div>
          )}
          <p className="status-message">{streamError || statusMessage}</p>
        </div>

        <div className="right">
          <div className="card green">
            <h2>Live Count</h2>
            <div className="big-text">{count}</div>
          </div>
          <div className="card orange">
            <h2>Camera Status</h2>
            <div className="big-text status-text">{cameraStatus}</div>
          </div>
        </div>
      </div>

      {/* ── upload section ── */}
      <div className="container upload-layout">
        <div className="upload-box">
          <h2>Upload Video</h2>

          <form className="upload-form" onSubmit={handleAnalyze}>

            {/* DROPDOWN — file picker இல்ல */}
            <select
              className="file-input"
              value={selectedVideo}
              onChange={handleVideoChange}
            >
              <option value="">-- Select a video --</option>
              {videos.length === 0 && (
                <option disabled>No videos in assets folder</option>
              )}
              {videos.map((v) => (
                <option key={v.name} value={v.name}>
                  {v.name}
                </option>
              ))}
            </select>

            <button
              type="submit"
              className="upload-button"
              disabled={isAnalyzing || !selectedVideo}
            >
              {isAnalyzing ? "Analyzing..." : "Upload & Count Bottles"}
            </button>
           <button
             type="button"
             className="upload-button"
             style={{ background: "#e53e3e", marginLeft: "8px" }}
             onClick={handleReset}
           >
              Reset
          </button>
          </form>

          {analysisError && <p className="error-message">{analysisError}</p>}

          {/* live detection stream */}
          {showLiveStream && (
            <>
              {analysisStreamError ? (
                <div className="empty-preview">{analysisStreamError}</div>
              ) : (
                <div className="live-preview-shell">
                  <img
                    src={analysisStreamUrl}
                    alt="live detection"
                    className="video-preview live-stream-preview"
                    onLoad={() => setAnalysisStreamError("")}
                    onError={() => setAnalysisStreamError("Live detection stream could not be loaded.")}
                  />
                  <div className="live-preview-badge">
                    <span>Live detection preview</span>
                    <strong>Total: {liveUploadStatus?.total_bottle_count ?? 0}</strong>
                  </div>
                </div>
              )}
              <p className="helper-text playback-note">
                {analysisProgress?.message || "Live detection is starting..."}
              </p>
            </>
          )}

          {/* completed — annotated video */}
          {!showLiveStream && analysisResult?.annotated_video_url && (
            <>
              <video
                key={analysisResult.annotated_video_url}
                ref={detectedVideoRef}
                src={`${API_BASE}${analysisResult.annotated_video_url}`}
                controls
                autoPlay
                muted
                playsInline
                preload="auto"
                loop
                className="video-preview"
              />
              <p className="helper-text playback-note">
                Detection is complete. Annotated replay with bounding boxes is ready here.
              </p>
            </>
          )}

          {/* nothing selected yet */}
          {!showLiveStream && !analysisResult && (
            <div className="empty-preview">
              Select a video from the dropdown and click analyze.
            </div>
          )}
        </div>

        {/* ── results ── */}
        <div className="results-box">
          <h2>Upload Result</h2>

          {isAnalysisRunning(analysisProgress) ? (
            <div className="results-grid">
              <div className="card green">
                <h3>Total Bottle Count</h3>
                <div className="big-text">{liveUploadStatus?.total_bottle_count ?? 0}</div>
              </div>
              <div className="card white-card">
                <h3>Playback Length</h3>
                <div className="metric-text">{formatSeconds(liveUploadStatus?.duration_seconds)}</div>
              </div>
            </div>
          ) : analysisResult ? (
            <>
              <div className="results-grid">
                <div className="card dark">
                  <h3>Total Bottle Count</h3>
                  <div className="big-text">{analysisResult.total_bottle_count}</div>
                </div>
                <div className="card white-card">
                  <h3>Video Duration</h3>
                  <div className="metric-text">{formatSeconds(analysisResult.duration_seconds)}</div>
                </div>
              </div>
              <div className="detected-frame-box">
                <h3>Counting Mode</h3>
                <p className="helper-text status-inline">
                  Total Bottle Count tracks bottles across frames, so the same bottle is counted once instead of repeating in every frame.
                </p>
              </div>
            </>
          ) : (
            <div className="empty-result">
              Upload a video and run analysis to see the bottle count summary here.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;