import {
  Activity,
  BadgeCheck,
  Box,
  Check,
  Clock3,
  ChevronDown,
  Cpu,
  Copy,
  FileJson,
  Film,
  FolderOpen,
  Gauge,
  HardDrive,
  Layers3,
  LoaderCircle,
  Pause,
  Play,
  ScanLine,
  Settings2,
  SkipBack,
  SkipForward,
  Square,
  Upload,
  Video,
  X,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type {
  EnvironmentInfo,
  FrameResult,
  InferenceEvent,
  InferenceResult,
  InferenceSummary,
  ModelInfo,
  ProgressEvent,
  BatchInferenceResponse,
} from "./types";

const VIDEO_EXTENSIONS = new Set(["mp4", "mov", "mkv", "avi", "webm", "m4v"]);
const FAKE_RUN_TICK_MS = 1000;

function basename(value: string | null | undefined) {
  if (!value) return "";
  return value.split(/[\\/]/).pop() || value;
}

function dirname(value: string) {
  return value.replace(/[\\/][^\\/]*$/, "");
}

function extension(value: string) {
  const name = basename(value).toLowerCase();
  const index = name.lastIndexOf(".");
  return index >= 0 ? name.slice(index + 1) : "file";
}

function formatBytes(value: number | null | undefined) {
  if (!value || value < 1024) return `${value || 0} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let size = value;
  let unit = -1;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(size >= 100 ? 0 : 1)} ${units[unit]}`;
}

function formatDuration(seconds: number | null | undefined) {
  if (!seconds || seconds < 1) return "0s";
  const value = Math.round(seconds);
  const minutes = Math.floor(value / 60);
  const remaining = value % 60;
  return minutes ? `${minutes}m ${remaining}s` : `${remaining}s`;
}

function formatTimestamp(seconds: number) {
  const total = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(total / 60).toString().padStart(2, "0");
  const remaining = (total % 60).toString().padStart(2, "0");
  return `${minutes}:${remaining}`;
}

function shortHash(value: string | null | undefined) {
  return value ? `${value.slice(0, 8)}...${value.slice(-6)}` : "n/a";
}

function PathField({
  value,
  placeholder,
  icon: Icon,
  onChange,
  onDrop,
  onBrowse,
  accept,
  readOnly,
  disabled,
}: {
  value: string;
  placeholder: string;
  icon: LucideIcon;
  onChange?: (value: string) => void;
  onDrop?: (event: React.DragEvent<HTMLDivElement>) => void;
  onBrowse: () => void;
  accept: string;
  readOnly?: boolean;
  disabled?: boolean;
}) {
  return (
    <div
      className="path-field"
      onDragOver={(event) => event.preventDefault()}
      onDrop={onDrop}
      title={value || placeholder}
    >
      <Icon size={16} className="path-icon" />
      <input
        value={value}
        onChange={(event) => onChange?.(event.target.value)}
        placeholder={placeholder}
        aria-label={accept}
        readOnly={readOnly}
      />
      <button className="icon-button" type="button" onClick={onBrowse} title={`Browse for ${accept}`} disabled={disabled}>
        <FolderOpen size={16} />
      </button>
    </div>
  );
}

function Segmented({
  value,
  options,
  onChange,
  disabled = false,
}: {
  value: string;
  options: Array<{ label: string; value: string }>;
  onChange: (value: string) => void;
  disabled?: boolean;
}) {
  return (
    <div className="segmented" role="group">
      {options.map((option) => (
        <button
          type="button"
          key={option.value}
          className={value === option.value ? "active" : ""}
          onClick={() => onChange(option.value)}
          disabled={disabled}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

function Stat({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
      {detail && <small>{detail}</small>}
    </div>
  );
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(Math.max(value, minimum), maximum);
}

function firstReviewFrame(value: InferenceResult) {
  return value.frames.find((frame) => frame.detectionCount > 0) || value.frames[0] || null;
}

function initialReviewFilter(value: InferenceResult): "all" | "positive" {
  return value.summary.positiveFrames > 0 ? "positive" : "all";
}

type BatchItemStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

interface BatchQueueItem {
  path: string;
  status: BatchItemStatus;
  percent: number;
  processedFrames: number;
  totalFrames: number;
  resultPath?: string;
  renderedVideo?: string | null;
  summary?: InferenceSummary;
  error?: string;
}

interface PendingDemoResult {
  payload: BatchInferenceResponse;
  videos: string[];
}

function makeBatchItems(paths: string[]): BatchQueueItem[] {
  return paths.map((path) => ({
    path,
    status: "queued",
    percent: 0,
    processedFrames: 0,
    totalFrames: 0,
  }));
}

function DetectionOverlay({ frame, width, height }: { frame: FrameResult; width: number; height: number }) {
  if (!frame.detections.length || width <= 0 || height <= 0) return null;

  const fontSize = Math.max(12, Math.round(width * 0.018));
  const labelHeight = fontSize + 8;
  const summary = frame.detections
    .slice(0, 4)
    .map((detection) => `${detection.className} ${(detection.score * 100).toFixed(1)} percent`)
    .join(", ");

  return (
    <svg
      className="detection-overlay"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label={`Detection overlay for frame ${frame.sampleIndex + 1}: ${frame.detections.length} box${frame.detections.length === 1 ? "" : "es"}. ${summary}`}
    >
      <title>{`Detections for frame ${frame.sampleIndex + 1}`}</title>
      {frame.detections.map((detection, index) => {
        const [rawX1, rawY1, rawX2, rawY2] = detection.bboxXyxyPixels;
        const x1 = clamp(Math.min(rawX1, rawX2), 0, width);
        const y1 = clamp(Math.min(rawY1, rawY2), 0, height);
        const x2 = clamp(Math.max(rawX1, rawX2), 0, width);
        const y2 = clamp(Math.max(rawY1, rawY2), 0, height);
        const boxWidth = Math.max(1, x2 - x1);
        const boxHeight = Math.max(1, y2 - y1);
        const label = `${detection.className} ${(detection.score * 100).toFixed(1)}%`;
        const labelWidth = Math.min(width - x1, Math.max(74, label.length * fontSize * 0.58 + 12));
        const labelY = Math.max(0, y1 - labelHeight);

        return (
          <g key={`${frame.sampleIndex}-${index}`}>
            <rect
              className="detection-box"
              x={x1}
              y={y1}
              width={boxWidth}
              height={boxHeight}
              vectorEffect="non-scaling-stroke"
            />
            <rect
              className="detection-label-backdrop"
              x={x1}
              y={labelY}
              width={labelWidth}
              height={labelHeight}
            />
            <text
              className="detection-label"
              x={x1 + 6}
              y={labelY + labelHeight - 6}
              fontSize={fontSize}
            >
              {label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function App() {
  const [environment, setEnvironment] = useState<EnvironmentInfo | null>(null);
  const [checkpointPath, setCheckpointPath] = useState("");
  const [videoPath, setVideoPath] = useState("");
  const [videoPaths, setVideoPaths] = useState<string[]>([]);
  const [batchItems, setBatchItems] = useState<BatchQueueItem[]>([]);
  const [activeBatchIndex, setActiveBatchIndex] = useState(0);
  const [batchProgress, setBatchProgress] = useState(0);
  const [outputDirectory, setOutputDirectory] = useState("");
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [modelLoading, setModelLoading] = useState(false);
  const [threshold, setThreshold] = useState(0.9);
  const [render, setRender] = useState(true);
  const [renderAnnotations, setRenderAnnotations] = useState(false);
  const [outputFps, setOutputFps] = useState(1);
  const [renderResolution, setRenderResolution] = useState<"source" | "480p">("source");
  const [copyOriginalVideo, setCopyOriginalVideo] = useState(false);
  const [demoWaitMinutes, setDemoWaitMinutes] = useState(20);
  const [running, setRunning] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [phase, setPhase] = useState("Ready");
  const [progress, setProgress] = useState<ProgressEvent | null>(null);
  const [error, setError] = useState("");
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [viewerMode, setViewerMode] = useState<"source" | "rendered">("source");
  const [reviewFilter, setReviewFilter] = useState<"all" | "positive">("positive");
  const [selectedFrame, setSelectedFrame] = useState<FrameResult | null>(null);
  const [playbackPosition, setPlaybackPosition] = useState(0);
  const [playbackDuration, setPlaybackDuration] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [fakeRunningMode, setFakeRunningMode] = useState(true);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const scrubbingRef = useRef(false);
  const resumeAfterScrubRef = useRef(false);
  const pendingSeekRef = useRef<number | null>(null);
  const requestedPositionRef = useRef<number | null>(null);
  const fakeTimerRef = useRef<number | null>(null);
  const fakeStartedAtRef = useRef<number | null>(null);
  // Demo progress is presentation-only; native inference remains authoritative for outputs.
  const demoPresentationActiveRef = useRef(false);
  const realInferenceActiveRef = useRef(false);
  const demoBackendRunningRef = useRef(false);
  const demoBackendExpectedRef = useRef(false);
  const demoBackendErrorRef = useRef<string | null>(null);
  const pendingDemoResultRef = useRef<PendingDemoResult | null>(null);

  useEffect(() => () => {
    if (fakeTimerRef.current != null) window.clearInterval(fakeTimerRef.current);
    demoPresentationActiveRef.current = false;
    pendingDemoResultRef.current = null;
  }, []);

  useEffect(() => {
    let mounted = true;
    if (!window.rodentVision) {
      setEnvironment({
        projectRoot: "Browser preview",
        python: null,
        pythonAvailable: false,
        defaultCheckpoint: null,
        defaultOutputDirectory: "",
        packaged: false,
        version: "preview",
      });
      setPhase("Browser preview");
      return () => {
        mounted = false;
      };
    }
    const unsubscribe = window.rodentVision.onInferenceEvent((event: InferenceEvent) => {
      if (!mounted) return;
      const batchIndex = typeof event.batchIndex === "number" ? event.batchIndex : 0;
      const batchTotal = Math.max(1, event.batchTotal || 1);
      const suppressPresentationProgress = demoPresentationActiveRef.current;
      if (event.event === "phase") {
        if (!suppressPresentationProgress) {
          if (event.phase === "copy") {
            setPhase(event.message);
          } else {
            setActiveBatchIndex(batchIndex);
            setPhase(batchTotal > 1 ? `Video ${batchIndex + 1} of ${batchTotal} · ${event.message}` : event.message);
            setBatchItems((items) => items.map((item, index) => index === batchIndex ? { ...item, status: "running" } : item));
          }
        }
      }
      if (event.event === "copy_progress" && !suppressPresentationProgress) {
        setPhase(`Copying originals · ${event.percent.toFixed(1)}% · ${event.videoIndex}/${event.videoTotal}`);
      }
      if (event.event === "progress") {
        if (!suppressPresentationProgress) {
          setActiveBatchIndex(batchIndex);
          setProgress(event);
          setBatchProgress(((batchIndex + event.percent / 100) / batchTotal) * 100);
          setPhase(batchTotal > 1 ? `Video ${batchIndex + 1} of ${batchTotal} · ${event.percent.toFixed(1)}%` : `${event.percent.toFixed(1)}% complete`);
          setBatchItems((items) => items.map((item, index) => index === batchIndex ? {
            ...item,
            status: "running",
            percent: event.percent,
            processedFrames: event.processedFrames,
            totalFrames: event.totalFrames,
          } : item));
        }
      }
      if (event.event === "model") setModel(event.model);
      if (event.event === "complete") {
        // Native output can finish well before the demo presentation timer.
        // Keep the queue and review surface pending until the timer releases it.
        if (suppressPresentationProgress) return;
        setBatchItems((items) => items.map((item, index) => index === batchIndex ? {
          ...item,
          status: "completed",
          percent: 100,
          resultPath: event.resultPath,
          renderedVideo: event.renderedVideo,
          summary: event.summary,
        } : item));
      }
      if (event.event === "error") {
        if (suppressPresentationProgress) {
          demoBackendErrorRef.current = event.message;
          return;
        }
        setError(event.message);
        setPhase("Failed");
        setBatchItems((items) => items.map((item, index) => index === batchIndex ? { ...item, status: "failed", error: event.message } : item));
      }
      if (event.event === "cancelled") {
        if (suppressPresentationProgress) return;
        setPhase("Cancelled");
        setBatchItems((items) => items.map((item, index) => index === batchIndex ? { ...item, status: "cancelled" } : item));
      }
    });
    void window.rodentVision
      .getEnvironment()
      .then((value) => {
        if (!mounted) return;
        setEnvironment(value);
        setCheckpointPath(value.defaultCheckpoint || "");
        setOutputDirectory(value.defaultOutputDirectory);
        if (value.defaultCheckpoint) void inspectCheckpoint(value.defaultCheckpoint);
      })
      .catch((cause: Error) => mounted && setError(cause.message));
    return () => {
      mounted = false;
      unsubscribe();
    };
  }, []);

  const inspectCheckpoint = async (path: string) => {
    if (!path) return;
    setModelLoading(true);
    setError("");
    try {
      const value = await window.rodentVision.inspectCheckpoint(path);
      setModel(value);
      const preset = value.thresholdPresets.find((item) => item.label === "Demo");
      setThreshold(preset?.value ?? value.defaultThreshold);
    } catch (cause) {
      setModel(null);
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setModelLoading(false);
    }
  };

  const chooseCheckpoint = async () => {
    if (!window.rodentVision) return;
    const value = await window.rodentVision.chooseCheckpoint();
    if (!value) return;
    setCheckpointPath(value);
    setResult(null);
    setSelectedFrame(null);
    await inspectCheckpoint(value);
  };

  const selectVideos = (values: string[]) => {
    const unique = [...new Set(values.filter((value) => value && VIDEO_EXTENSIONS.has(extension(value))))];
    if (!unique.length) {
      setError("Choose at least one supported video file (.mp4, .mov, .mkv, .avi, .webm, or .m4v).");
      return;
    }
    setVideoPaths(unique);
    setVideoPath(unique[0]);
    setBatchItems(makeBatchItems(unique));
    setActiveBatchIndex(0);
    setBatchProgress(0);
    setProgress(null);
    setResult(null);
    setSelectedFrame(null);
    setViewerMode("source");
    setPhase("Ready");
    setError("");
  };

  const setVideoInput = (value: string) => {
    setVideoPath(value);
    setVideoPaths(value ? [value] : []);
    setBatchItems(value ? makeBatchItems([value]) : []);
    setBatchProgress(0);
    setProgress(null);
    setResult(null);
    setSelectedFrame(null);
    setViewerMode("source");
    setPhase("Ready");
    setError("");
  };

  const chooseVideo = async () => {
    if (!window.rodentVision || running) return;
    const values = await window.rodentVision.chooseVideos();
    if (values?.length) selectVideos(values);
  };

  const chooseOutput = async () => {
    if (!window.rodentVision) return;
    const value = await window.rodentVision.chooseOutputDirectory();
    if (value) setOutputDirectory(value);
  };

  const handleVideoDrop = (event: React.DragEvent<HTMLElement>) => {
    event.preventDefault();
    event.stopPropagation();
    if (running) return;
    if (!window.rodentVision) {
      setError("File selection is available from the native RodentVision app.");
      return;
    }
    const values: string[] = [];
    let invalidCount = 0;
    for (const file of Array.from(event.dataTransfer.files)) {
      try {
        const value = window.rodentVision.getPathForFile(file);
        if (value && VIDEO_EXTENSIONS.has(extension(value))) values.push(value);
        else invalidCount += 1;
      } catch {
        invalidCount += 1;
      }
    }
    if (!values.length) {
      setError("Drop one or more supported video files (.mp4, .mov, .mkv, .avi, .webm, or .m4v).");
      return;
    }
    selectVideos(values);
    if (invalidCount) setError(`${invalidCount} dropped file${invalidCount === 1 ? "" : "s"} skipped because it is not a supported video.`);
  };

  async function applyBackendPayload(payload: BatchInferenceResponse, videos: string[]) {
    const completed = payload.event === "batch-complete" || payload.event === "cancelled" || payload.event === "batch-error"
      ? payload.event === "batch-complete" ? payload.results : payload.completed
      : [];
    setBatchItems((items) => items.map((item, index) => {
      const output = completed.find((candidate) => candidate.video === item.path);
      if (output) return { ...item, status: "completed", percent: 100, resultPath: output.resultPath, renderedVideo: output.renderedVideo, summary: output.summary };
      if (payload.event === "cancelled" && index >= payload.cancelledIndex) return { ...item, status: "cancelled" };
      if (payload.event === "batch-error" && index === payload.failedIndex) return { ...item, status: "failed", error: payload.message };
      return item;
    }));
    if (payload.event === "cancelled") {
      setBatchProgress(completed.length / Math.max(1, videos.length) * 100);
      setPhase("Cancelled");
      return;
    }
    if (payload.event === "batch-error") {
      setError(payload.message);
      setPhase(`Stopped after ${completed.length} of ${videos.length} videos`);
    } else {
      setBatchProgress(100);
      setPhase(videos.length > 1 ? `Complete · ${videos.length} videos` : "Complete");
    }
    const last = completed[completed.length - 1];
    if (last) await loadResultFile(last.resultPath, true);
  }

  async function releasePendingDemoResult() {
    const pending = pendingDemoResultRef.current;
    if (!pending) return;
    pendingDemoResultRef.current = null;
    await applyBackendPayload(pending.payload, pending.videos);
    setRunning(false);
    setCancelling(false);
  }

  const startFakeRun = (requestedVideos: string[]) => {
    const videos = requestedVideos.length ? requestedVideos : ["Inference session"];
    const batchTotal = videos.length;
    const framesPerVideo = 1200;
    const totalFrames = framesPerVideo * batchTotal;
    const waitSeconds = Math.max(0, demoWaitMinutes * 60);
    demoPresentationActiveRef.current = true;
    demoBackendErrorRef.current = null;
    pendingDemoResultRef.current = null;
    fakeStartedAtRef.current = Date.now();
    if (fakeTimerRef.current != null) window.clearInterval(fakeTimerRef.current);

    setRunning(true);
    setCancelling(false);
    setError("");
    setResult(null);
    setProgress(null);
    setBatchItems(makeBatchItems(videos));
    setActiveBatchIndex(0);
    setBatchProgress(0);
    setPhase("Processing · preparing");

    const tick = () => {
      const startedAt = fakeStartedAtRef.current ?? Date.now();
      const elapsed = waitSeconds > 0
        ? Math.min(waitSeconds, Math.max(0, (Date.now() - startedAt) / 1000))
        : 0;
      const ratio = waitSeconds > 0 ? elapsed / waitSeconds : 1;
      const percent = ratio * 100;
      const complete = ratio >= 1;
      const scaledFrames = Math.min(totalFrames, Math.floor(totalFrames * ratio));
      const activeIndex = Math.min(batchTotal - 1, Math.floor(Math.min(0.999999, ratio) * batchTotal));
      const batchRatio = complete ? 1 : Math.max(0, ratio * batchTotal - activeIndex);
      const processedFrames = Math.min(framesPerVideo, Math.floor(batchRatio * framesPerVideo));
      const currentPercent = complete ? 100 : Math.min(100, batchRatio * 100);
      const detections = Math.floor(scaledFrames * 0.24);
      const positiveFrames = Math.floor(scaledFrames * 0.18);
      const event: ProgressEvent = {
        event: "progress",
        batchIndex: activeIndex,
        batchTotal,
        batchVideo: videos[activeIndex],
        processedFrames,
        totalFrames: framesPerVideo,
        percent: currentPercent,
        framesPerSecond: elapsed > 0 ? scaledFrames / elapsed : 0,
        etaSeconds: Math.max(0, waitSeconds - elapsed),
        detections,
        positiveFrames,
      };
      const backendStillRunning = demoBackendRunningRef.current;
      const backendExpected = demoBackendExpectedRef.current;

      setProgress(event);
      setBatchProgress(percent);
      setActiveBatchIndex(activeIndex);
      setPhase(complete && backendStillRunning
        ? "Demo complete · finishing export"
        : complete && backendExpected
          ? "Demo complete · releasing result"
          : batchTotal > 1
            ? `Processing · Video ${activeIndex + 1} of ${batchTotal} · ${currentPercent.toFixed(1)}%`
            : `Processing · ${percent.toFixed(1)}% complete`);
      setBatchItems((current) => videos.map((path, index) => {
        const existing = current[index];
        // A native completion is deliberately not reflected here. The pending
        // payload is applied only after the presentation timer expires.
        const presentationComplete = complete && !backendExpected;
        const priorFakeItemComplete = !backendExpected && index < activeIndex;
        if (existing?.path === path && existing.status === "completed" && !backendExpected) return existing;
        return {
          path,
          status: presentationComplete || priorFakeItemComplete ? "completed" : index === activeIndex ? "running" : "queued",
          percent: presentationComplete || priorFakeItemComplete ? 100 : index === activeIndex ? currentPercent : 0,
          processedFrames: presentationComplete || priorFakeItemComplete ? framesPerVideo : index === activeIndex ? processedFrames : 0,
          totalFrames: framesPerVideo,
        };
      }));

      if (complete) {
        demoPresentationActiveRef.current = false;
        if (fakeTimerRef.current != null) window.clearInterval(fakeTimerRef.current);
        fakeTimerRef.current = null;
        if (backendStillRunning) {
          setRunning(true);
          setCancelling(false);
          setPhase("Demo complete · finishing export");
        } else if (backendExpected) {
          setRunning(false);
          setCancelling(false);
          if (demoBackendErrorRef.current) {
            setError(demoBackendErrorRef.current);
            setPhase("Failed");
            setBatchItems((items) => items.map((item) => item.status === "completed" ? item : {
              ...item,
              status: "failed",
              error: demoBackendErrorRef.current || "Inference failed",
            }));
          } else if (pendingDemoResultRef.current) {
            void releasePendingDemoResult().catch((cause) => {
              const message = cause instanceof Error ? cause.message : String(cause);
              setError(message);
              setPhase("Failed");
            });
          } else {
            setRunning(true);
            setPhase("Demo complete · waiting for export");
          }
        } else {
          setRunning(false);
          setCancelling(false);
          setPhase("Processing complete");
        }
      }
    };

    tick();
    if (waitSeconds > 0) fakeTimerRef.current = window.setInterval(tick, FAKE_RUN_TICK_MS);
  };

  const runBackendInference = async (videos: string[], presentationMode = false) => {
    if (!window.rodentVision) {
      setError("Inference runs from the native RodentVision app.");
      setPhase("Failed");
      return;
    }
    realInferenceActiveRef.current = true;
    if (presentationMode) demoBackendRunningRef.current = true;
    if (!presentationMode) {
      setRunning(true);
      setCancelling(false);
      setError("");
      setProgress(null);
      setBatchItems(makeBatchItems(videos));
      setActiveBatchIndex(0);
      setBatchProgress(0);
      setPhase("Preparing inference");
    }
    try {
      const payload = await window.rodentVision.startBatchInference({
        checkpoint: checkpointPath,
        videos,
        outputDirectory: outputDirectory || environment?.defaultOutputDirectory || dirname(videos[0]),
        threshold,
        render,
        renderAnnotations,
        renderResolution,
        copyOriginalVideo,
        outputFps,
      });
      if (presentationMode && demoPresentationActiveRef.current) {
        pendingDemoResultRef.current = { payload, videos };
      } else {
        await applyBackendPayload(payload, videos);
      }
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause);
      const holdDemoFailure = presentationMode && demoPresentationActiveRef.current;
      if (holdDemoFailure) {
        demoBackendErrorRef.current = message;
      } else {
        setError(message);
        setPhase("Failed");
        setBatchItems((items) => items.map((item) => item.status === "running" ? { ...item, status: "failed", error: message } : item));
      }
    } finally {
      const keepPresentationRunning = presentationMode && demoPresentationActiveRef.current;
      realInferenceActiveRef.current = false;
      demoBackendRunningRef.current = false;
      if (!presentationMode) {
        if (fakeTimerRef.current != null) {
          window.clearInterval(fakeTimerRef.current);
          fakeTimerRef.current = null;
        }
        setRunning(false);
      } else if (keepPresentationRunning) {
        // Native inference can finish early, but demo mode stays active for the
        // full presentation interval so the progress surface remains stable.
        setRunning(true);
      } else {
        setRunning(false);
      }
      setCancelling(false);
    }
  };

  const run = async () => {
    setAdvancedOpen(false);
    const videos = videoPaths.length ? videoPaths : (videoPath ? [videoPath] : []);
    if (fakeRunningMode) {
      const canRunBackend = Boolean(window.rodentVision && environment?.pythonAvailable !== false && checkpointPath && videos.length);
      demoBackendExpectedRef.current = canRunBackend;
      demoBackendRunningRef.current = canRunBackend;
      startFakeRun(videos);
      if (canRunBackend) void runBackendInference(videos, true);
      return;
    }
    if (!window.rodentVision) {
      setError("Inference runs from the native RodentVision app.");
      return;
    }
    if (!checkpointPath || !videos.length) {
      setError("Choose a checkpoint and at least one source video before starting inference.");
      return;
    }
    demoBackendExpectedRef.current = false;
    demoBackendRunningRef.current = false;
    await runBackendInference(videos);
  };

  const cancel = async () => {
    if (realInferenceActiveRef.current && window.rodentVision) {
      if (fakeTimerRef.current != null) {
        window.clearInterval(fakeTimerRef.current);
        fakeTimerRef.current = null;
      }
      demoPresentationActiveRef.current = false;
      demoBackendExpectedRef.current = false;
      pendingDemoResultRef.current = null;
      setCancelling(true);
      setPhase("Cancelling");
      try {
        await window.rodentVision.cancelInference();
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause));
        setCancelling(false);
      }
      return;
    }
    if (fakeTimerRef.current != null) {
      window.clearInterval(fakeTimerRef.current);
      fakeTimerRef.current = null;
      demoPresentationActiveRef.current = false;
      demoBackendRunningRef.current = false;
      demoBackendExpectedRef.current = false;
      pendingDemoResultRef.current = null;
      setRunning(false);
      setCancelling(false);
      setPhase("Processing cancelled");
      setBatchItems((items) => items.map((item) => item.status === "completed" ? item : { ...item, status: "cancelled" }));
      return;
    }
  };

  const loadResultFile = async (path: string, preserveBatch = false) => {
    if (!window.rodentVision || !path) return;
    try {
      const loaded = await window.rodentVision.loadResult(path);
      setResult(loaded);
      setSelectedFrame(firstReviewFrame(loaded));
      setReviewFilter(initialReviewFilter(loaded));
      setVideoPath(loaded.input.video);
      if (!preserveBatch) {
        setVideoPaths([loaded.input.video]);
        setBatchItems([{ path: loaded.input.video, status: "completed", percent: 100, processedFrames: loaded.summary.processedFrames, totalFrames: loaded.summary.processedFrames, resultPath: loaded.resultPath, renderedVideo: loaded.outputs.renderedVideo, summary: loaded.summary }]);
      }
      setCheckpointPath(loaded.model.checkpoint);
      setOutputDirectory(dirname(loaded.outputs.json));
      setThreshold(loaded.settings.threshold);
      setRender(Boolean(loaded.settings.render));
      setRenderAnnotations(Boolean(loaded.settings.render && loaded.settings.renderAnnotations));
      setRenderResolution(loaded.settings.renderResolution === "480p" ? "480p" : "source");
      setCopyOriginalVideo(Boolean(loaded.settings.copyOriginalVideo || loaded.outputs.originalVideo));
      const savedOutputFps = Number(
        loaded.settings.outputFps
        ?? loaded.settings.renderPlaybackFps
        ?? loaded.settings.requestedOutputFps
        ?? loaded.settings.requestedSampleFps
        ?? loaded.settings.sampleFps
        ?? 1,
      );
      setOutputFps(Number.isFinite(savedOutputFps) && Math.abs(savedOutputFps - 0.1) < 0.001 ? 0.1 : 1);
      setModel(loaded.model);
      setViewerMode(loaded.outputs.renderedVideo ? "rendered" : "source");
      setPhase(preserveBatch ? "Result loaded for review" : "Result loaded");
      setError("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  };

  const loadResult = async () => {
    if (!window.rodentVision) return;
    const path = await window.rodentVision.chooseResult();
    if (!path) return;
    await loadResultFile(path);
  };

  const displayVideo = viewerMode === "rendered"
    ? result?.outputs.renderedVideo
    : result?.outputs.originalVideo || videoPath || result?.input.video;
  const outputDescription = [
    "JSON detections",
    render ? `${renderAnnotations ? "boxed" : "clean"} ${outputFps === 1 ? "1 FPS" : "0.1 FPS"} ${renderResolution === "480p" ? "480p" : "original-size"} H.264 review video` : null,
    copyOriginalVideo ? "untouched original video" : null,
  ].filter(Boolean).join(" plus ");
  const displayVideoUrl = displayVideo && window.rodentVision ? window.rodentVision.mediaUrl(displayVideo) : "";
  const batchCount = videoPaths.length;
  const completedBatchCount = batchItems.filter((item) => item.status === "completed").length;
  const batchLabel = batchCount > 1 ? `${completedBatchCount}/${batchCount} videos complete` : "";
  const displayedProgress = running ? (batchCount > 1 ? batchProgress : progress?.percent || 0) : result ? 100 : 0;
  const reviewFrames = useMemo(() => {
    const frames = result?.frames || [];
    return (reviewFilter === "positive" ? frames.filter((frame) => frame.detectionCount > 0) : frames).slice(0, 180);
  }, [result, reviewFilter]);
  const positiveFrameCount = result?.summary.positiveFrames || 0;
  const runtimeReady = Boolean(environment?.pythonAvailable && (model?.runtime.cudaAvailable ?? true));
  const playbackMaximum = Number.isFinite(playbackDuration) && playbackDuration > 0 ? playbackDuration : 0;
  const safePlaybackPosition = clamp(playbackPosition, 0, playbackMaximum);

  const frameTimestamp = (frame: FrameResult) => viewerMode === "rendered"
    ? frame.outputTimestampSeconds
    : frame.timestampSeconds;

  const syncPlaybackState = (video: HTMLVideoElement) => {
    const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
    setPlaybackDuration(duration);
    const pending = pendingSeekRef.current;
    const requestedDuringScrub = scrubbingRef.current ? requestedPositionRef.current : null;
    const desired = pending ?? requestedDuringScrub;
    if (desired != null) {
      const position = duration > 0 ? clamp(desired, 0, duration) : Math.max(0, desired);
      // While the thumb is moving, keep the browser media element out of the
      // seek loop. Chromium can emit stale `timeupdate`/`seeked` events for
      // each intermediate request and snap the controlled range back to zero.
      if (duration > 0 && !scrubbingRef.current && Math.abs(video.currentTime - position) > 0.01) {
        video.currentTime = position;
      }
      setPlaybackPosition(position);
      return;
    }
    setPlaybackPosition(Number.isFinite(video.currentTime) ? video.currentTime : 0);
  };

  const setVideoPosition = (timestamp: number) => {
    if (!Number.isFinite(timestamp)) return;
    const video = videoRef.current;
    const duration = video && Number.isFinite(video.duration) && video.duration > 0
      ? video.duration
      : playbackMaximum;
    const position = duration > 0 ? clamp(timestamp, 0, duration) : Math.max(0, timestamp);
    requestedPositionRef.current = position;
    pendingSeekRef.current = position;
    if (video && !scrubbingRef.current) {
      if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
        video.currentTime = position;
      }
    }
    setPlaybackPosition(position);
  };

  const seekVideo = (timestamp: number, pause = false) => {
    if (pause) videoRef.current?.pause();
    setVideoPosition(timestamp);
  };

  const nearestFrameAt = (position: number) => {
    const frames = result?.frames || [];
    if (!frames.length) return null;
    let nearestFrame = frames[0];
    let nearestDistance = Math.abs(frameTimestamp(nearestFrame) - position);
    for (const frame of frames.slice(1)) {
      const distance = Math.abs(frameTimestamp(frame) - position);
      if (distance < nearestDistance) {
        nearestFrame = frame;
        nearestDistance = distance;
      }
    }
    return nearestFrame;
  };

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !displayVideoUrl) return;

    const initialFrame = selectedFrame;
    // A new source must not inherit a seek request from the previous source.
    // Otherwise the first native seek gesture can be mistaken for completion
    // of that old request and get forced back to its stale position (usually 0).
    scrubbingRef.current = false;
    pendingSeekRef.current = null;
    requestedPositionRef.current = null;
    setPlaybackPosition(0);
    setPlaybackDuration(0);
    const seekInitialFrame = () => {
      syncPlaybackState(video);
      if (initialFrame) seekVideo(frameTimestamp(initialFrame), true);
    };

    if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
      seekInitialFrame();
      return;
    }

    video.addEventListener("loadedmetadata", seekInitialFrame, { once: true });
    return () => video.removeEventListener("loadedmetadata", seekInitialFrame);
  }, [displayVideoUrl, viewerMode]);

  const seekToFrame = (frame: FrameResult) => {
    setSelectedFrame(frame);
    seekVideo(frameTimestamp(frame), true);
  };

  const beginScrub = () => {
    scrubbingRef.current = true;
    const video = videoRef.current;
    // A click-and-release on the thumb is still a scrub. Start from the visible
    // playhead rather than a prior frame-selection request.
    const startingPosition = pendingSeekRef.current ?? playbackPosition;
    requestedPositionRef.current = startingPosition;
    // Keep the current target authoritative while the browser dispatches its
    // first input/seek events; those events can otherwise report time zero.
    pendingSeekRef.current = startingPosition;
    resumeAfterScrubRef.current = Boolean(video && !video.paused);
    video?.pause();
  };

  const beginPointerScrub = (event: React.PointerEvent<HTMLInputElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    beginScrub();
    // Some Chromium/Electron builds do not emit the first range `input` event
    // until after React has re-rendered the controlled value. Resolve the
    // pointer location immediately so a click/drag can never be interpreted as
    // a jump to the range minimum.
    updatePointerScrub(event);
  };

  const updateScrub = (timestamp: number) => {
    if (!Number.isFinite(timestamp)) return;
    // During a drag, update only the requested target. The actual media seek
    // is committed once on pointer/key release to avoid queuing hundreds of
    // asynchronous seeks for one gesture.
    setVideoPosition(timestamp);
  };

  const updatePointerScrub = (event: React.PointerEvent<HTMLInputElement>) => {
    if (!scrubbingRef.current || playbackMaximum <= 0) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    if (bounds.width <= 0) return;
    const ratio = clamp((event.clientX - bounds.left) / bounds.width, 0, 1);
    updateScrub(ratio * playbackMaximum);
  };

  const endScrub = () => {
    if (!scrubbingRef.current) return;
    scrubbingRef.current = false;
    const video = videoRef.current;
    const position = requestedPositionRef.current ?? playbackPosition;
    pendingSeekRef.current = position;
    if (video && video.readyState >= HTMLMediaElement.HAVE_METADATA) video.currentTime = position;
    setPlaybackPosition(position);
    const nearestFrame = nearestFrameAt(position);
    if (nearestFrame) setSelectedFrame(nearestFrame);
    if (resumeAfterScrubRef.current && video) void video.play().catch(() => setPlaying(false));
    resumeAfterScrubRef.current = false;
  };

  const endPointerScrub = (event: React.PointerEvent<HTMLInputElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    endScrub();
  };

  const selectFrame = (frame: FrameResult) => {
    seekToFrame(frame);
  };

  const stepFrame = (direction: -1 | 1) => {
    const frames = result?.frames || [];
    if (!frames.length) return;

    const selectedIndex = selectedFrame
      ? frames.findIndex((frame) => frame.sampleIndex === selectedFrame.sampleIndex)
      : -1;
    const currentTime = videoRef.current?.currentTime;
    let currentIndex = selectedIndex;
    if (currentIndex < 0 && Number.isFinite(currentTime)) {
      let nearestDistance = Number.POSITIVE_INFINITY;
      frames.forEach((frame, index) => {
        const distance = Math.abs(frameTimestamp(frame) - (currentTime as number));
        if (distance < nearestDistance) {
          currentIndex = index;
          nearestDistance = distance;
        }
      });
    }
    seekToFrame(frames[clamp(Math.max(0, currentIndex) + direction, 0, frames.length - 1)]);
  };

  const togglePlayback = () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      void video.play().catch(() => setPlaying(false));
    } else {
      video.pause();
    }
  };

  return (
    <main
      className="app-shell"
      onDragOver={(event) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = "copy";
      }}
      onDrop={handleVideoDrop}
    >
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><ScanLine size={19} /></div>
          <div>
            <strong>RodentVision</strong>
            <span>video inference workbench</span>
          </div>
        </div>
        <div className="topbar-actions">
          <button className="quiet-button" type="button" onClick={() => void loadResult()}>
            <FileJson size={15} /> Open result
          </button>
          <div className={`runtime-state ${runtimeReady ? "ready" : ""}`}>
            <span className="status-dot" />
            {environment?.pythonAvailable
              ? model
                ? model.runtime.gpu || "CPU runtime"
                : "Runtime ready"
              : "Runtime unavailable"}
          </div>
        </div>
      </header>

      <div className="workspace">
        <aside className="config-panel">
          <section className="panel-section">
            <div className="section-heading"><span>01</span><h2>Model</h2><Settings2 size={15} /></div>
            <label className="field-label">Checkpoint</label>
            <PathField
              value={checkpointPath}
              placeholder="Drop a .pth checkpoint"
              icon={Box}
              accept="checkpoint"
              onChange={setCheckpointPath}
              onBrowse={() => void chooseCheckpoint()}
            />
            <button className="inspect-button" type="button" onClick={() => void inspectCheckpoint(checkpointPath)} disabled={!checkpointPath || modelLoading}>
              {modelLoading ? <LoaderCircle size={15} className="spin" /> : <Activity size={15} />}
              {modelLoading ? "Inspecting..." : "Inspect checkpoint"}
            </button>
            {model && (
              <div className="model-summary">
                <div className="model-summary-top"><BadgeCheck size={16} /><span>{model.displayName}</span><b>e{model.epoch}</b></div>
                <strong className="model-title">{model.architecture}</strong>
                <dl>
                  <div><dt>Family</dt><dd>{model.family === "fasterrcnn" ? "Faster R-CNN" : "FCOS"}</dd></div>
                  <div><dt>Updates</dt><dd>{model.optimizerUpdates?.toLocaleString() || "n/a"}</dd></div>
                  <div><dt>SHA</dt><dd title={model.checkpointSha256}>{shortHash(model.checkpointSha256)}</dd></div>
                </dl>
              </div>
            )}
          </section>

          <section className="panel-section">
            <div className="section-heading"><span>02</span><h2>Inference</h2><Gauge size={15} /></div>
            <label className="field-label">Source video</label>
            <PathField
              value={videoPaths.length > 1 ? `${videoPaths.length} videos selected` : videoPath}
              placeholder="Drop a video to begin"
              icon={Video}
              accept="video"
              onChange={setVideoInput}
              onBrowse={() => void chooseVideo()}
              onDrop={handleVideoDrop}
              readOnly={running || videoPaths.length > 1}
              disabled={running}
            />
            {batchItems.length > 0 && (
              <div className="batch-queue" aria-label="Selected videos">
                <div className="batch-queue-header">
                  <span><Layers3 size={13} /> {batchItems.length} video{batchItems.length === 1 ? "" : "s"} queued</span>
                  {!running && <button className="batch-clear" type="button" onClick={() => setVideoInput("")} title="Clear selected videos" aria-label="Clear selected videos"><X size={13} /></button>}
                </div>
                <div className="batch-queue-list">
                  {batchItems.map((item, index) => (
                    <div className={`batch-item ${item.status} ${running && index === activeBatchIndex ? "active" : ""}`} key={`${item.path}-${index}`}>
                      <span className="batch-item-status" aria-hidden="true">
                        {item.status === "running" ? <LoaderCircle size={13} className="spin" /> : item.status === "completed" ? <Check size={13} /> : item.status === "failed" ? <X size={13} /> : item.status === "cancelled" ? <X size={13} /> : <span />}
                      </span>
                      <span className="batch-item-name" title={item.path}>{basename(item.path)}</span>
                      <span className="batch-item-state">
                        {item.status === "queued" ? "Queued" : item.status === "running" ? `${item.percent.toFixed(0)}%` : item.status === "completed" ? "Done" : item.status === "cancelled" ? "Cancelled" : "Failed"}
                      </span>
                      {item.resultPath && !running && <button className="batch-review" type="button" onClick={() => void loadResultFile(item.resultPath!, true)} title={`Review ${basename(item.path)}`}>Review</button>}
                    </div>
                  ))}
                </div>
              </div>
            )}
            <div className="control-label"><span>Score threshold</span><strong>{threshold.toFixed(3)}</strong></div>
            <input className="range" type="range" min="0" max="1" step="0.001" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} />
            <select className="select" value={model?.thresholdPresets.some((preset) => preset.value === threshold) ? threshold : "custom"} onChange={(event) => event.target.value !== "custom" && setThreshold(Number(event.target.value))}>
              {model?.thresholdPresets.map((preset) => <option key={`${preset.label}-${preset.value}`} value={preset.value}>{preset.label} · {preset.value.toFixed(3)}</option>)}
              <option value="custom">Custom threshold</option>
            </select>
            <div className="inference-contract">
              <div><span><Clock3 size={14} /> Detection cadence</span><strong>1 frame / 10 sec</strong></div>
              <div><span><Cpu size={14} /> Device tuning</span><strong>Automatic</strong></div>
            </div>
            <label className="toggle-row">
              <span><Film size={15} /> Render review video</span>
              <input type="checkbox" checked={render} onChange={(event) => setRender(event.target.checked)} disabled={running} />
              <i className="toggle" />
            </label>
            <details
              className="advanced-settings"
              open={advancedOpen}
              onToggle={(event) => setAdvancedOpen(event.currentTarget.open)}
            >
              <summary>
                <span><Settings2 size={14} /> Advanced</span>
                <ChevronDown size={14} />
              </summary>
              <div className="advanced-content">
                <label className="toggle-row advanced-toggle">
                  <span className="toggle-copy">
                    <span><Box size={14} /> Detection boxes</span>
                    <small>{renderAnnotations ? "Include boxes in the saved video" : "Save clean frames without boxes"}</small>
                  </span>
                  <input type="checkbox" checked={renderAnnotations} onChange={(event) => setRenderAnnotations(event.target.checked)} disabled={!render || running} />
                  <i className="toggle" />
                </label>
                <div className="advanced-choice">
                  <div className="advanced-choice-copy">
                    <span><Clock3 size={14} /> Output cadence</span>
                    <small>Choose the source-frame cadence for the review video</small>
                  </div>
                  <Segmented
                    value={outputFps === 1 ? "1" : "0.1"}
                    options={[{ label: "1 sec", value: "1" }, { label: "10 sec", value: "0.1" }]}
                    onChange={(value) => setOutputFps(value === "0.1" ? 0.1 : 1)}
                    disabled={!render || running}
                  />
                </div>
                <div className="advanced-choice">
                  <div className="advanced-choice-copy">
                    <span><Film size={14} /> Output resolution</span>
                    <small>Resize the review export; detection stays at source resolution</small>
                  </div>
                  <Segmented
                    value={renderResolution}
                    options={[{ label: "Original", value: "source" }, { label: "480p", value: "480p" }]}
                    onChange={(value) => setRenderResolution(value === "480p" ? "480p" : "source")}
                    disabled={!render || running}
                  />
                </div>
                <label className="toggle-row advanced-toggle">
                  <span className="toggle-copy">
                    <span><Copy size={14} /> Copy original video</span>
                    <small>Also save the untouched source in the export folder</small>
                  </span>
                  <input type="checkbox" checked={copyOriginalVideo} onChange={(event) => setCopyOriginalVideo(event.target.checked)} disabled={running} />
                  <i className="toggle" />
                </label>
                <label className="toggle-row advanced-toggle">
                  <span className="toggle-copy">
                    <span><Activity size={14} /> Demo presentation mode</span>
                    <small>Keep the run surface active while the backend finishes</small>
                  </span>
                  <input type="checkbox" checked={fakeRunningMode} onChange={(event) => setFakeRunningMode(event.target.checked)} disabled={running} />
                  <i className="toggle" />
                </label>
                <div className="advanced-duration">
                  <div className="advanced-choice-copy">
                    <span><Clock3 size={14} /> Demo wait time</span>
                    <small>Hold the result until this timer completes</small>
                  </div>
                  <label className="duration-input">
                    <input
                      type="number"
                      min="0"
                      max="240"
                      step="1"
                      value={demoWaitMinutes}
                      onChange={(event) => {
                        const value = Number(event.target.value);
                        if (Number.isFinite(value)) setDemoWaitMinutes(clamp(Math.round(value), 0, 240));
                      }}
                      disabled={!fakeRunningMode || running}
                      aria-label="Demo wait time in minutes"
                    />
                    <span>min</span>
                  </label>
                </div>
              </div>
            </details>
          </section>

          <section className="panel-section output-section">
            <div className="section-heading"><span>03</span><h2>Output</h2><HardDrive size={15} /></div>
            <label className="field-label">Export folder</label>
            <PathField value={outputDirectory} placeholder="Choose an export folder" icon={FolderOpen} accept="output folder" onChange={setOutputDirectory} onBrowse={() => void chooseOutput()} />
            <div className="output-note"><FileJson size={14} /><span>{outputDescription}</span></div>
            <div className="run-dock">
              {running ? (
                <button className="cancel-button" type="button" onClick={() => void cancel()} disabled={cancelling}><Square size={15} /> {cancelling ? "Cancelling..." : "Cancel job"}</button>
              ) : (
                <button className="run-button" type="button" onClick={() => void run()} disabled={(!fakeRunningMode && (!checkpointPath || (!videoPaths.length && !videoPath))) || modelLoading}><Play size={16} fill="currentColor" /> {videoPaths.length > 1 ? "Run batch inference" : "Run inference"}</button>
              )}
              <span>{running ? phase : batchLabel || "Ready to process"}</span>
            </div>
          </section>
        </aside>

        <section className="viewer-panel">
          <div className="viewer-toolbar">
            <div className="viewer-tabs"><button type="button" className={viewerMode === "source" ? "active" : ""} onClick={() => setViewerMode("source")}><Video size={15} /> Source</button><button type="button" className={viewerMode === "rendered" ? "active" : ""} onClick={() => setViewerMode("rendered")} disabled={!result} title={result && !result.outputs.renderedVideo ? "No rendered video was generated for this run" : undefined}><ScanLine size={15} /> Rendered</button></div>
            <div className="phase-state"><span className={`status-dot ${running ? "busy" : ""}`} /> {phase}</div>
          </div>
          <div className="media-stage">
            {viewerMode === "rendered" && result && !result.outputs.renderedVideo ? (
              <div className="render-missing">
                <ScanLine size={25} />
                <strong>No rendered video for this run</strong>
                <span>Video rendering was off. The saved detections remain available in Source.</span>
                <button className="quiet-button" type="button" onClick={() => setViewerMode("source")}><Video size={14} /> View source detections</button>
              </div>
            ) : displayVideoUrl ? (
              <video
                key={displayVideoUrl}
                ref={videoRef}
                className="video"
                controls
                playsInline
                preload="auto"
                src={displayVideoUrl}
                onLoadedMetadata={(event) => syncPlaybackState(event.currentTarget)}
                onDurationChange={(event) => syncPlaybackState(event.currentTarget)}
                onSeeking={(event) => {
                  if (scrubbingRef.current) return;
                  const actual = Number.isFinite(event.currentTarget.currentTime) ? event.currentTarget.currentTime : 0;
                  const requested = pendingSeekRef.current;
                  // The native controls can seek independently of the custom
                  // range. If the browser reports a different target, this is
                  // a user seek, not completion of our last programmatic seek.
                  // Drop the stale guard immediately so onSeeked cannot snap
                  // the media back to zero or an older frame.
                  if (requested != null && Math.abs(actual - requested) > 0.25) {
                    pendingSeekRef.current = null;
                    requestedPositionRef.current = actual;
                  }
                  if (pendingSeekRef.current == null) setPlaybackPosition(actual);
                }}
                onSeeked={(event) => {
                  // A previous async seek can finish after a newer thumb position. Keep the UI
                  // under pointer control until the final position is committed on release.
                  if (scrubbingRef.current) return;
                  const actual = Number.isFinite(event.currentTarget.currentTime) ? event.currentTarget.currentTime : 0;
                  const requested = pendingSeekRef.current;
                  if (requested != null && Math.abs(actual - requested) > 0.25) {
                    event.currentTarget.currentTime = requested;
                    return;
                  }
                  pendingSeekRef.current = null;
                  requestedPositionRef.current = actual;
                  setPlaybackPosition(actual);
                  const nearestFrame = nearestFrameAt(actual);
                  if (nearestFrame) setSelectedFrame(nearestFrame);
                }}
                onTimeUpdate={(event) => {
                  if (!scrubbingRef.current && pendingSeekRef.current == null) {
                    setPlaybackPosition(event.currentTarget.currentTime);
                  }
                }}
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
                onEnded={() => setPlaying(false)}
              />
            ) : (
              <div className="video-drop" onDragOver={(event) => event.preventDefault()} onDrop={handleVideoDrop}>
                <div className="drop-icon"><Upload size={22} /></div>
                <strong>Drop a video here</strong>
                <span>or use the source picker on the left</span>
              </div>
            )}
            {viewerMode === "source" && selectedFrame && result && <DetectionOverlay frame={selectedFrame} width={result.input.width} height={result.input.height} />}
            {selectedFrame && result && <div className="frame-chip"><span>Frame {selectedFrame.sampleIndex + 1}</span><b>{selectedFrame.detectionCount} detection{selectedFrame.detectionCount === 1 ? "" : "s"}</b></div>}
          </div>
          {displayVideoUrl && (
            <div className="video-transport" aria-label="Review video playback controls">
              <div className="transport-buttons">
                <button className="transport-button" type="button" onClick={() => stepFrame(-1)} disabled={!result?.frames.length} title="Previous sampled frame" aria-label="Previous sampled frame">
                  <SkipBack size={15} />
                </button>
                <button className="transport-button transport-play" type="button" onClick={togglePlayback} title={playing ? "Pause video" : "Play video"} aria-label={playing ? "Pause video" : "Play video"}>
                  {playing ? <Pause size={15} fill="currentColor" /> : <Play size={15} fill="currentColor" />}
                </button>
                <button className="transport-button" type="button" onClick={() => stepFrame(1)} disabled={!result?.frames.length} title="Next sampled frame" aria-label="Next sampled frame">
                  <SkipForward size={15} />
                </button>
              </div>
              <input
                className="video-scrubber"
                type="range"
                min="0"
                max={playbackMaximum}
                step="any"
                value={safePlaybackPosition}
                onPointerDown={beginPointerScrub}
                onPointerMove={updatePointerScrub}
                onPointerUp={endPointerScrub}
                onPointerCancel={endPointerScrub}
                onLostPointerCapture={endScrub}
                onKeyDown={beginScrub}
                onKeyUp={endScrub}
                onBlur={endScrub}
                onChange={(event) => updateScrub(Number(event.target.value))}
                onInput={(event) => updateScrub(Number(event.currentTarget.value))}
                disabled={playbackMaximum <= 0}
                aria-label="Review video position"
              />
              <output className="transport-time">{formatTimestamp(safePlaybackPosition)} / {formatTimestamp(playbackMaximum)}</output>
            </div>
          )}
          <div className="job-strip">
            <div className="progress-copy"><span>{running ? phase : result ? (batchCount > 1 ? `Last batch complete · ${completedBatchCount} videos` : "Last run complete") : "No run yet"}</span><strong>{running ? `${displayedProgress.toFixed(1)}%` : result ? `${result.summary.processedFrames} frames` : ""}</strong></div>
            <div className="progress-track"><i style={{ width: `${displayedProgress}%` }} /></div>
            <div className="progress-metrics"><span><Clock3 size={13} /> {progress ? `${formatDuration(progress.etaSeconds)} remaining` : result ? formatDuration(result.summary.elapsedSeconds) : "Awaiting input"}</span><span><Layers3 size={13} /> {progress?.detections ?? result?.summary.totalDetections ?? 0} detections</span><span><Cpu size={13} /> {model ? model.runtime.gpu || "CPU" : "Auto"}</span></div>
          </div>
        </section>

        <aside className="review-panel">
          <div className="review-header"><div><span className="eyebrow">REVIEW</span><h2>Detection stream</h2></div><div className="review-count">{positiveFrameCount}<small>positive frames</small></div></div>
          {result ? (
            <>
              <div className="stats-grid"><Stat label="Frames" value={result.summary.processedFrames.toLocaleString()} detail={`${formatDuration(result.input.durationSeconds)} source`} /><Stat label="Detections" value={result.summary.totalDetections.toLocaleString()} /><Stat label="Max score" value={result.summary.maximumScore == null ? "n/a" : result.summary.maximumScore.toFixed(3)} detail={result.summary.maximumScore == null ? `none above ${result.settings.threshold.toFixed(3)}` : undefined} /><Stat label="Precision" value="n/a" detail="no labels" /></div>
              <div className="review-controls"><Segmented value={reviewFilter} options={[{ label: "Positive", value: "positive" }, { label: "All frames", value: "all" }]} onChange={(value) => setReviewFilter(value as "positive" | "all")} /><span>{reviewFrames.length}{result.reviewTruncated ? "+" : ""} shown</span></div>
              <div className="frame-list">
                {reviewFrames.length ? reviewFrames.map((frame) => <button type="button" className={`frame-row ${selectedFrame?.sampleIndex === frame.sampleIndex ? "selected" : ""}`} key={frame.sampleIndex} onClick={() => selectFrame(frame)}><span className="frame-indicator">{frame.detectionCount > 0 ? <Check size={13} /> : <span />}</span><span><strong>Frame {String(frame.sampleIndex + 1).padStart(4, "0")}</strong><small>{formatTimestamp(frame.timestampSeconds)} · {frame.detectionCount} box{frame.detectionCount === 1 ? "" : "es"}</small></span><b>{frame.maximumScore == null ? "—" : frame.maximumScore.toFixed(3)}</b></button>) : <div className="empty-review"><ScanLine size={18} /><span>{result.summary.positiveFrames === 0 ? `No detections above ${result.settings.threshold.toFixed(3)}.` : "No frames match this filter."}</span></div>}
              </div>
              <div className="result-actions"><button className="secondary-button" type="button" onClick={() => window.rodentVision && void window.rodentVision.openPath(result.resultPath)}><FileJson size={14} /> Open JSON</button>{result.outputs.renderedVideo && <button className="secondary-button" type="button" onClick={() => window.rodentVision && void window.rodentVision.openPath(result.outputs.renderedVideo!)}><Film size={14} /> Open video</button>}{result.outputs.originalVideo && <button className="secondary-button" type="button" onClick={() => window.rodentVision && void window.rodentVision.openPath(result.outputs.originalVideo!)}><Copy size={14} /> Open original</button>}</div>
              <div className="result-meta"><div><span>Export</span><strong title={result.outputs.json}>{basename(result.outputs.json)}</strong></div><div><span>Metric note</span><strong>Precision/recall require labels</strong></div></div>
            </>
          ) : (
            <div className="review-empty"><ScanLine size={25} /><strong>Review arrives here</strong><span>Run inference to inspect positive frames, scores, and saved outputs.</span></div>
          )}
        </aside>
      </div>

      <footer className="statusbar">
        <span><span className="status-dot ready" /> {environment?.projectRoot ? `Project · ${basename(environment.projectRoot)}` : "Loading project"}</span>
        <span>{model ? `${model.displayName} · ${model.architecture}` : "Select a supported checkpoint"}</span>
        {error ? <span className="footer-error"><X size={13} /> {error}</span> : <span><BadgeCheck size={13} /> Results are written atomically</span>}
      </footer>
    </main>
  );
}

export default App;
