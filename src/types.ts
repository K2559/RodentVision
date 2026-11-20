export interface ThresholdPreset {
  label: string;
  value: number;
  detail: string;
}

export interface ModelInfo {
  checkpoint: string;
  checkpointSha256: string;
  checkpointBytes: number;
  schema: string;
  family: "fcos" | "fasterrcnn";
  modelId: string;
  displayName: string;
  architecture: string;
  architectureContract: string;
  epoch: number;
  arm: string;
  optimizerUpdates: number | null;
  parameterSha256: string | null;
  createdAtUtc: string | null;
  defaultThreshold: number;
  thresholdPresets: ThresholdPreset[];
  input: {
    minimumSize: number;
    maximumSize: number;
  };
  runtime: {
    torch: string;
    cudaAvailable: boolean;
    cudaVersion: string | null;
    gpu: string | null;
  };
  projectRoot: string;
}

export interface EnvironmentInfo {
  projectRoot: string;
  python: string | null;
  pythonAvailable: boolean;
  defaultCheckpoint: string | null;
  defaultOutputDirectory: string;
  packaged: boolean;
  version: string;
}

export interface Detection {
  classId: number;
  className: string;
  score: number;
  bboxXyxyPixels: [number, number, number, number];
  bboxXyxyNormalized: [number, number, number, number];
  areaPixels: number;
}

export interface FrameResult {
  sampleIndex: number;
  sourceFrameIndex: number;
  timestampSeconds: number;
  outputFrameIndex?: number;
  outputTimestampSeconds: number;
  detectionCount: number;
  maximumScore: number | null;
  detections: Detection[];
}

export interface InferenceSummary {
  processedFrames: number;
  positiveFrames: number;
  negativeFrames: number;
  totalDetections: number;
  maximumScore: number | null;
  meanDetectionScore: number | null;
  elapsedSeconds: number;
  averageProcessedFps: number;
  precision: null;
  recall: null;
  accuracyMetricsAvailable: false;
}

export interface InferenceResult {
  schema: string;
  createdAtUtc: string;
  status: "completed";
  input: {
    video: string;
    videoBytes: number;
    width: number;
    height: number;
    fps: number;
    frames: number;
    durationSeconds: number;
  };
  model: ModelInfo;
  settings: {
    threshold: number;
    detectionFps?: number;
    detectionIntervalSeconds?: number;
    outputFps?: number;
    requestedOutputFps?: number;
    outputIntervalSeconds?: number;
    sampleFps?: number;
    requestedSampleFps?: number;
    samplingIntervalSeconds?: number;
    renderPlaybackFps?: number | null;
    renderAnnotations?: boolean;
    renderResolution?: "source" | "480p" | null;
    copyOriginalVideo?: boolean;
    batchSize: number;
    batchMode: "auto" | "manual";
    device: string;
    render: boolean;
  };
  outputs: {
    json: string;
    renderedVideo: string | null;
    originalVideo?: string | null;
    renderedVideoInfo: Record<string, unknown> | null;
  };
  summary: InferenceSummary;
  runtime: {
    python: string;
    torch: string;
    opencv: string;
    cuda: string | null;
    gpu: string | null;
    peakGpuMemoryGiB: number | null;
    ffmpeg: string | null;
    encoder: string | null;
  };
  frames: FrameResult[];
  metricNote: string;
  reviewTruncated?: boolean;
  resultPath: string;
}

export interface InferenceBatchMeta {
  batchIndex?: number;
  batchTotal?: number;
  batchVideo?: string;
}

export interface ProgressEvent extends InferenceBatchMeta {
  event: "progress";
  processedFrames: number;
  totalFrames: number;
  percent: number;
  framesPerSecond: number;
  etaSeconds: number;
  detections: number;
  positiveFrames: number;
}

export interface CopyProgressEvent extends InferenceBatchMeta {
  event: "copy_progress";
  videoIndex: number;
  videoTotal: number;
  copiedBytes: number;
  totalBytes: number;
  percent: number;
  video: string;
}

export type InferenceEvent =
  | ProgressEvent
  | CopyProgressEvent
  | ({ event: "phase"; phase: string; message: string; totalFrames?: number } & InferenceBatchMeta)
  | ({
      event: "model";
      model: ModelInfo;
      device: string;
      batchSize?: number;
      batchMode?: string;
    } & InferenceBatchMeta)
  | ({ event: "complete"; resultPath: string; renderedVideo: string | null; originalVideo?: string | null; summary: InferenceSummary } & InferenceBatchMeta)
  | ({ event: "cancelled"; message: string } & InferenceBatchMeta)
  | ({ event: "error"; message: string; errorType?: string } & InferenceBatchMeta);

export interface StartInferenceOptions {
  checkpoint: string;
  video: string;
  outputDirectory: string;
  threshold: number;
  render: boolean;
  renderAnnotations: boolean;
  renderResolution?: "source" | "480p";
  copyOriginalVideo: boolean;
  outputFps: number;
  sampleFps?: number;
}

export interface StartBatchInferenceOptions {
  checkpoint: string;
  videos: string[];
  outputDirectory: string;
  threshold: number;
  render: boolean;
  renderAnnotations: boolean;
  renderResolution?: "source" | "480p";
  copyOriginalVideo: boolean;
  outputFps: number;
  sampleFps?: number;
}

export interface BatchInferenceOutput {
  video: string;
  resultPath: string;
  renderedVideo: string | null;
  originalVideo?: string | null;
  summary: InferenceSummary;
}

export type BatchInferenceResponse =
  | { event: "batch-complete"; results: BatchInferenceOutput[] }
  | { event: "cancelled"; message: string; completed: BatchInferenceOutput[]; cancelledIndex: number }
  | { event: "batch-error"; message: string; completed: BatchInferenceOutput[]; failedIndex: number };

export type LitteringCategory = "pedestrian_littering" | "vehicle_littering" | "bird_feeding" | "dog_presence";

export interface GeminiEvent {
  timestampSeconds: number;
  category: LitteringCategory;
  score: number;
}

export interface GeminiResult {
  model: string;
  events: GeminiEvent[];
}

export interface RodentVisionBridge {
  geminiConfigured(): Promise<boolean>;
  analyzeWithGemini(options: { video: string; duration: number; key: string }): Promise<GeminiResult | null>;
  cancelGemini(): Promise<void>;
  onGeminiPhase(callback: (message: string) => void): () => void;
  saveLitteringReport(format: "csv" | "pdf", csv?: string): Promise<string | null>;
  getEnvironment(): Promise<EnvironmentInfo>;
  chooseCheckpoint(): Promise<string | null>;
  chooseVideo(): Promise<string | null>;
  chooseVideos(): Promise<string[] | null>;
  chooseResult(): Promise<string | null>;
  chooseOutputDirectory(): Promise<string | null>;
  inspectCheckpoint(checkpoint: string): Promise<ModelInfo>;
  startInference(options: StartInferenceOptions): Promise<
    | {
        event: "complete";
        resultPath: string;
        renderedVideo: string | null;
        summary: InferenceSummary;
      }
    | { event: "cancelled"; message: string }
  >;
  startBatchInference(options: StartBatchInferenceOptions): Promise<BatchInferenceResponse>;
  cancelInference(): Promise<boolean>;
  loadResult(resultPath: string): Promise<InferenceResult>;
  showItemInFolder(targetPath: string): Promise<boolean>;
  openPath(targetPath: string): Promise<boolean>;
  getPathForFile(file: File): string;
  mediaUrl(targetPath: string): string;
  onInferenceEvent(callback: (event: InferenceEvent) => void): () => void;
}
