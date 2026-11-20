const {
  app,
  BrowserWindow,
  dialog,
  ipcMain,
  net,
  protocol,
  shell,
} = require("electron");
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const { pathToFileURL } = require("node:url");

protocol.registerSchemesAsPrivileged([
  {
    scheme: "rodent-media",
    privileges: {
      secure: true,
      standard: true,
      supportFetchAPI: true,
      stream: true,
    },
  },
]);

const SOURCE_ROOT = path.resolve(__dirname, "..");
const KNOWN_PROJECT_ROOT = path.resolve(SOURCE_ROOT, "..");
const DEFAULT_CHECKPOINT_PARTS = [
  "training_outputs",
  "pilot21_screened_quota_comparison",
  "run5",
  "control",
  "checkpoint_epoch_02.pth",
];
const DEFAULT_OUTPUT_FPS = 1;
const ALLOWED_OUTPUT_FPS = [0.1, 1];
const DEFAULT_RENDER_RESOLUTION = "source";
const ALLOWED_RENDER_RESOLUTIONS = new Set(["source", "480p"]);
const AUTO_BATCH_SIZE = 0;
const BACKEND_START_TIMEOUT_MS = 30_000;
const BACKEND_ACTIVITY_TIMEOUT_MS = 180_000;
const VIDEO_EXTENSIONS = new Set([".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"]);

let mainWindow = null;
let activeInference = null;

function terminateProcessTree(child) {
  if (!child || child.exitCode !== null || !child.pid) return;
  if (process.platform === "win32") {
    const taskkill = path.join(process.env.SystemRoot || "C:\\Windows", "System32", "taskkill.exe");
    const killer = spawn(taskkill, ["/PID", String(child.pid), "/T", "/F"], {
      windowsHide: true,
      stdio: "ignore",
    });
    killer.unref();
    return;
  }
  child.kill("SIGKILL");
}

function existingProjectRoot() {
  const candidates = [
    process.env.RODENTVISION_PROJECT_ROOT,
    KNOWN_PROJECT_ROOT,
  ].filter(Boolean);
  return candidates.find((candidate) =>
    fs.existsSync(path.join(candidate, "training_scripts", "pilot18_two_stage_v21.py")),
  ) || FALLBACK_PROJECT_ROOT;
}

function backendPath() {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, "app.asar.unpacked", "backend", "inference_runner.py");
  }
  return path.join(SOURCE_ROOT, "backend", "inference_runner.py");
}

function pythonPath() {
  const projectRoot = existingProjectRoot();
  const candidates = [
    process.env.RODENTVISION_PYTHON,
    path.resolve(
      projectRoot,
      "..",
      "shared",
      "workspace_core",
      "train_v6_venv",
      "Scripts",
      "python.exe",
    ),
    "python.exe",
  ].filter(Boolean);
  return candidates.find((candidate) => candidate === "python.exe" || fs.existsSync(candidate));
}

function requirePath(value, label, { mustExist = true, directory = false } = {}) {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new TypeError(`${label} is required`);
  }
  const resolved = path.resolve(value);
  if (mustExist) {
    const stats = fs.statSync(resolved);
    if (directory ? !stats.isDirectory() : !stats.isFile()) {
      throw new TypeError(`${label} has the wrong file type: ${resolved}`);
    }
  }
  return resolved;
}

function samePath(first, second) {
  const normalize = process.platform === "win32"
    ? (value) => path.resolve(value).toLowerCase()
    : (value) => path.resolve(value);
  return normalize(first) === normalize(second);
}

function numberInRange(value, label, minimum, maximum) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < minimum || parsed > maximum) {
    throw new TypeError(`${label} must be between ${minimum} and ${maximum}`);
  }
  return parsed;
}

function outputFpsOption(value) {
  const parsed = Number(value ?? DEFAULT_OUTPUT_FPS);
  if (!Number.isFinite(parsed) || !ALLOWED_OUTPUT_FPS.some((candidate) => Math.abs(candidate - parsed) < 1e-9)) {
    throw new TypeError("Output cadence must be either 1 frame per second or 1 frame per 10 seconds");
  }
  return parsed;
}

function renderResolutionOption(value) {
  const selected = value ?? DEFAULT_RENDER_RESOLUTION;
  if (typeof selected !== "string" || !ALLOWED_RENDER_RESOLUTIONS.has(selected)) {
    throw new TypeError("Render resolution must be either source or 480p");
  }
  return selected;
}

function runBackend(
  argumentsList,
  {
    sender = null,
    exclusive = false,
    eventMeta = {},
    startupPhase = "startup",
    startupMessage = "Starting inference runtime",
  } = {},
) {
  return new Promise((resolve, reject) => {
    if (exclusive && activeInference) {
      reject(new Error("An inference job is already running"));
      return;
    }
    const runtime = pythonPath();
    const runner = backendPath();
    if (!runtime || !fs.existsSync(runner)) {
      reject(new Error("RodentVision could not locate its Python runtime or inference runner"));
      return;
    }

    const child = spawn(runtime, ["-u", runner, ...argumentsList], {
      cwd: existingProjectRoot(),
      windowsHide: true,
      shell: false,
      env: {
        ...process.env,
        PYTHONUTF8: "1",
        PYTHONUNBUFFERED: "1",
        RODENTVISION_PROJECT_ROOT: existingProjectRoot(),
      },
    });
    child.rodentVisionCancelRequested = false;
    if (exclusive) activeInference = child;
    sender?.send("inference:event", {
      event: "phase",
      phase: startupPhase,
      message: startupMessage,
      ...eventMeta,
    });

    let stdoutBuffer = "";
    let stderrBuffer = "";
    let lastPayload = null;
    let backendError = null;
    let settled = false;
    let watchdog = null;

    const clearWatchdog = () => {
      if (watchdog) clearTimeout(watchdog);
      watchdog = null;
    };
    const releaseActiveInference = () => {
      if (exclusive && activeInference === child) activeInference = null;
    };
    const resolveOnce = (payload) => {
      if (settled) return;
      settled = true;
      clearWatchdog();
      releaseActiveInference();
      resolve(payload);
    };
    const rejectOnce = (error) => {
      if (settled) return;
      settled = true;
      clearWatchdog();
      releaseActiveInference();
      reject(error);
    };
    const armWatchdog = (timeout, message) => {
      if (!exclusive || settled) return;
      clearWatchdog();
      watchdog = setTimeout(() => {
        const payload = { event: "error", errorType: "BackendTimeout", message };
        backendError = message;
        sender?.send("inference:event", { ...payload, ...eventMeta });
        terminateProcessTree(child);
        rejectOnce(new Error(message));
      }, timeout);
    };

    armWatchdog(
      BACKEND_START_TIMEOUT_MS,
      "The Python inference runtime did not respond within 30 seconds",
    );

    const consumeLine = (line) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      try {
        const payload = JSON.parse(trimmed);
        lastPayload = payload;
        if (payload.event === "error") backendError = payload.message;
        sender?.send("inference:event", { ...payload, ...eventMeta });
        if (!["complete", "cancelled", "error"].includes(payload.event)) {
          const activity = payload.message || payload.event;
          armWatchdog(
            BACKEND_ACTIVITY_TIMEOUT_MS,
            `Inference stopped responding during: ${activity}`,
          );
        }
      } catch {
        stderrBuffer += `${trimmed}\n`;
      }
    };

    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdoutBuffer += chunk;
      const lines = stdoutBuffer.split(/\r?\n/);
      stdoutBuffer = lines.pop() || "";
      lines.forEach(consumeLine);
    });
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk) => {
      stderrBuffer = `${stderrBuffer}${chunk}`.slice(-24_000);
    });
    child.on("error", (error) => {
      rejectOnce(error);
    });
    child.on("exit", (code, signal) => {
      if (stdoutBuffer) consumeLine(stdoutBuffer);
      if (settled) return;
      if (child.rodentVisionCancelRequested) {
        const payload = { event: "cancelled", message: "Inference cancelled" };
        sender?.send("inference:event", { ...payload, ...eventMeta });
        resolveOnce(payload);
        return;
      }
      if (code === 0) {
        resolveOnce(lastPayload);
        return;
      }
      const detail = backendError || stderrBuffer.trim() || `process exited with code ${code}`;
      rejectOnce(new Error(signal ? `${detail} (${signal})` : detail));
    });
  });
}

async function stageOriginalVideos(sender, videos, outputDirectory) {
  const args = ["stage-originals", "--output-dir", outputDirectory];
  for (const video of videos) args.push("--video", video);
  const payload = await runBackend(args, {
    sender,
    exclusive: true,
    startupPhase: "copy",
    startupMessage: `Preparing to copy ${videos.length} original video${videos.length === 1 ? "" : "s"}`,
  });
  if (payload?.event === "cancelled") return payload;
  if (!payload || payload.event !== "staged" || !Array.isArray(payload.originals)) {
    throw new Error("Original videos were not staged before inference");
  }
  if (payload.originals.length !== videos.length) {
    throw new Error("The staged original-video count does not match the selected batch");
  }
  const originals = payload.originals.map((entry, index) => {
    if (!entry || typeof entry !== "object") {
      throw new Error(`Original video ${index + 1} has no staging record`);
    }
    const source = requirePath(entry.video, `Staged source video ${index + 1}`);
    if (!samePath(source, videos[index])) {
      throw new Error(`Original video ${index + 1} was staged out of order`);
    }
    const copied = requirePath(entry.originalVideo, `Copied original video ${index + 1}`);
    if (!samePath(path.dirname(copied), outputDirectory)) {
      throw new Error(`Copied original video ${index + 1} is outside the export folder`);
    }
    return copied;
  });
  return { event: "staged", originals };
}

function registerIpc() {
  ipcMain.handle("environment:get", async () => {
    const projectRoot = existingProjectRoot();
    const checkpoint = path.join(projectRoot, ...DEFAULT_CHECKPOINT_PARTS);
    return {
      projectRoot,
      python: pythonPath(),
      pythonAvailable: Boolean(pythonPath()),
      defaultCheckpoint: fs.existsSync(checkpoint) ? checkpoint : null,
      defaultOutputDirectory: path.join(projectRoot, "training_outputs", "rodent_vision_exports"),
      packaged: app.isPackaged,
      version: app.getVersion(),
    };
  });

  ipcMain.handle("dialog:checkpoint", async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
      title: "Select model checkpoint",
      properties: ["openFile"],
      filters: [{ name: "PyTorch checkpoint", extensions: ["pth", "pt"] }],
    });
    return result.canceled ? null : result.filePaths[0];
  });

  ipcMain.handle("dialog:video", async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
      title: "Select source video",
      properties: ["openFile"],
      filters: [
        { name: "Video", extensions: ["mp4", "mov", "mkv", "avi", "webm", "m4v"] },
      ],
    });
    return result.canceled ? null : result.filePaths[0];
  });

  ipcMain.handle("dialog:videos", async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
      title: "Select source videos",
      properties: ["openFile", "multiSelections"],
      filters: [
        { name: "Video", extensions: ["mp4", "mov", "mkv", "avi", "webm", "m4v"] },
      ],
    });
    return result.canceled ? null : result.filePaths;
  });

  ipcMain.handle("dialog:result", async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
      title: "Open RodentVision result",
      properties: ["openFile"],
      filters: [{ name: "RodentVision JSON", extensions: ["json"] }],
    });
    return result.canceled ? null : result.filePaths[0];
  });

  ipcMain.handle("dialog:output-directory", async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
      title: "Select output folder",
      properties: ["openDirectory", "createDirectory"],
    });
    return result.canceled ? null : result.filePaths[0];
  });

  ipcMain.handle("checkpoint:inspect", async (_event, checkpoint) => {
    const checkpointPath = requirePath(checkpoint, "Checkpoint");
    const payload = await runBackend(["inspect", "--checkpoint", checkpointPath]);
    if (!payload || payload.event !== "model_info") {
      throw new Error("The checkpoint inspector returned no model metadata");
    }
    return payload.model;
  });

  ipcMain.handle("inference:start", async (event, options) => {
    if (!options || typeof options !== "object") throw new TypeError("Inference options are required");
    const checkpoint = requirePath(options.checkpoint, "Checkpoint");
    const video = requirePath(options.video, "Video");
    if (!VIDEO_EXTENSIONS.has(path.extname(video).toLowerCase())) {
      throw new TypeError("The selected input is not a supported video file");
    }
    const outputDirectory = requirePath(options.outputDirectory, "Output folder", {
      mustExist: false,
      directory: true,
    });
    const threshold = numberInRange(options.threshold, "Threshold", 0, 1);
    const outputFps = outputFpsOption(options.outputFps ?? options.sampleFps);
    const renderResolution = renderResolutionOption(options.renderResolution);
    let stagedOriginal = null;
    if (options.copyOriginalVideo === true) {
      const staging = await stageOriginalVideos(event.sender, [video], outputDirectory);
      if (staging.event === "cancelled") return staging;
      stagedOriginal = staging.originals[0];
    }
    const args = [
      "run",
      "--checkpoint",
      checkpoint,
      "--video",
      video,
      "--output-dir",
      outputDirectory,
      "--threshold",
      String(threshold),
      "--output-fps",
      String(outputFps),
      "--render-resolution",
      renderResolution,
      "--batch-size",
      String(AUTO_BATCH_SIZE),
      options.renderAnnotations === true ? "--render-annotations" : "--no-render-annotations",
      "--no-copy-original",
      options.render === false ? "--no-render" : "--render",
    ];
    if (stagedOriginal) args.push("--staged-original", stagedOriginal);
    const payload = await runBackend(args, { sender: event.sender, exclusive: true });
    if (payload?.event === "cancelled") return payload;
    if (!payload || payload.event !== "complete") {
      throw new Error("Inference ended without a completion record");
    }
    return payload;
  });

  ipcMain.handle("inference:start-batch", async (event, options) => {
    if (!options || typeof options !== "object") throw new TypeError("Batch inference options are required");
    const checkpoint = requirePath(options.checkpoint, "Checkpoint");
    if (!Array.isArray(options.videos) || options.videos.length === 0) {
      throw new TypeError("At least one source video is required");
    }
    if (options.videos.length > 100) {
      throw new TypeError("A batch can contain at most 100 videos");
    }
    const videos = options.videos.map((value, index) => {
      const video = requirePath(value, `Video ${index + 1}`);
      if (!VIDEO_EXTENSIONS.has(path.extname(video).toLowerCase())) {
        throw new TypeError(`Video ${index + 1} is not a supported video file`);
      }
      return video;
    });
    const outputDirectory = requirePath(options.outputDirectory, "Output folder", {
      mustExist: false,
      directory: true,
    });
    const threshold = numberInRange(options.threshold, "Threshold", 0, 1);
    const outputFps = outputFpsOption(options.outputFps ?? options.sampleFps);
    const renderResolution = renderResolutionOption(options.renderResolution);
    const completed = [];
    let stagedOriginals = Array(videos.length).fill(null);
    if (options.copyOriginalVideo === true) {
      let staging;
      try {
        staging = await stageOriginalVideos(event.sender, videos, outputDirectory);
      } catch (cause) {
        const message = `Copying original videos failed before inference: ${cause instanceof Error ? cause.message : String(cause)}`;
        event.sender.send("inference:event", { event: "error", errorType: "BatchCopyError", message });
        return { event: "batch-error", message, completed, failedIndex: 0 };
      }
      if (staging.event === "cancelled") {
        return { event: "cancelled", message: "Inference cancelled", completed, cancelledIndex: 0 };
      }
      stagedOriginals = staging.originals;
    }
    for (let index = 0; index < videos.length; index += 1) {
      const video = videos[index];
      const args = [
        "run",
        "--checkpoint",
        checkpoint,
        "--video",
        video,
        "--output-dir",
        outputDirectory,
        "--threshold",
        String(threshold),
        "--output-fps",
        String(outputFps),
        "--render-resolution",
        renderResolution,
        "--batch-size",
        String(AUTO_BATCH_SIZE),
        options.renderAnnotations === true ? "--render-annotations" : "--no-render-annotations",
        "--no-copy-original",
        options.render === false ? "--no-render" : "--render",
      ];
      if (stagedOriginals[index]) args.push("--staged-original", stagedOriginals[index]);
      const eventMeta = {
        batchIndex: index,
        batchTotal: videos.length,
        batchVideo: video,
      };
      let payload;
      try {
        payload = await runBackend(args, { sender: event.sender, exclusive: true, eventMeta });
      } catch (cause) {
        const message = `Video ${index + 1} (${path.basename(video)}) failed: ${cause instanceof Error ? cause.message : String(cause)}`;
        event.sender.send("inference:event", { event: "error", errorType: "BatchVideoError", message, ...eventMeta });
        return { event: "batch-error", message, completed, failedIndex: index };
      }
      if (payload?.event === "cancelled") {
        return { event: "cancelled", message: "Inference cancelled", completed, cancelledIndex: index };
      }
      if (!payload || payload.event !== "complete") {
        const message = `Video ${index + 1} (${path.basename(video)}) ended without a completion record`;
        event.sender.send("inference:event", { event: "error", errorType: "BatchCompletionError", message, ...eventMeta });
        return { event: "batch-error", message, completed, failedIndex: index };
      }
      completed.push({
        video,
        resultPath: payload.resultPath,
        renderedVideo: payload.renderedVideo,
        originalVideo: payload.originalVideo,
        summary: payload.summary,
      });
    }
    return { event: "batch-complete", results: completed };
  });

  ipcMain.handle("inference:cancel", async () => {
    const child = activeInference;
    if (!child) return false;
    child.rodentVisionCancelRequested = true;
    terminateProcessTree(child);
    return true;
  });

  ipcMain.handle("result:load", async (_event, resultPath) => {
    const resolved = requirePath(resultPath, "Result JSON");
    const payload = JSON.parse(await fs.promises.readFile(resolved, "utf8"));
    if (payload.schema !== "rodent-vision-inference/1.0" || !Array.isArray(payload.frames)) {
      throw new Error("This is not a RodentVision result file");
    }
    const maximumFrames = 50_000;
    return {
      ...payload,
      frames: payload.frames.slice(0, maximumFrames),
      reviewTruncated: payload.frames.length > maximumFrames,
      resultPath: resolved,
    };
  });

  ipcMain.handle("shell:show-item", async (_event, targetPath) => {
    const resolved = requirePath(targetPath, "Output");
    shell.showItemInFolder(resolved);
    return true;
  });

  ipcMain.handle("shell:open-path", async (_event, targetPath) => {
    const resolved = requirePath(targetPath, "Output");
    const message = await shell.openPath(resolved);
    if (message) throw new Error(message);
    return true;
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1580,
    height: 960,
    minWidth: 1080,
    minHeight: 700,
    backgroundColor: "#edf2f0",
    title: "RodentVision",
    autoHideMenuBar: true,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  mainWindow.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  mainWindow.webContents.on("will-navigate", (event, navigationUrl) => {
    if (navigationUrl !== mainWindow.webContents.getURL()) event.preventDefault();
  });
  mainWindow.once("ready-to-show", () => mainWindow.show());
  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  if (process.env.VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(process.env.VITE_DEV_SERVER_URL);
  } else {
    mainWindow.loadFile(path.join(SOURCE_ROOT, "dist", "index.html"));
  }

  const screenshotPath = process.env.RODENTVISION_SCREENSHOT_PATH;
  if (screenshotPath) {
    mainWindow.webContents.once("did-finish-load", () => {
      setTimeout(async () => {
        const image = await mainWindow.webContents.capturePage();
        fs.mkdirSync(path.dirname(screenshotPath), { recursive: true });
        fs.writeFileSync(screenshotPath, image.toPNG());
        app.quit();
      }, Number(process.env.RODENTVISION_SCREENSHOT_DELAY_MS || 8000));
    });
  }
}

const hasLock = app.requestSingleInstanceLock();
if (!hasLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });

  app.whenReady().then(() => {
    protocol.handle("rodent-media", (request) => {
      try {
        const requestUrl = new URL(request.url);
        const targetPath = requestUrl.searchParams.get("path");
        if (!targetPath) return new Response("Missing media path", { status: 400 });
        const resolved = requirePath(targetPath, "Media");
        if (!VIDEO_EXTENSIONS.has(path.extname(resolved).toLowerCase())) {
          return new Response("Unsupported media type", { status: 415 });
        }
        // Preserve media seek headers. Without the Range request, Chromium can
        // only read the file from byte zero and native/custom timelines snap
        // back to the beginning after a drag.
        const mediaHeaders = {};
        // Electron versions expose protocol headers either as a plain object
        // or as a Fetch Headers instance. Support both so Range survives the
        // custom media URL in every packaged runtime.
        const requestHeaderEntries = typeof request.headers?.entries === "function"
          ? Array.from(request.headers.entries())
          : Object.entries(request.headers || {});
        for (const [header, value] of requestHeaderEntries) {
          if (/^(range|if-range|if-none-match|if-modified-since|accept|cache-control)$/i.test(header)) {
            mediaHeaders[header] = value;
          }
        }
        return net.fetch(pathToFileURL(resolved).toString(), { headers: mediaHeaders });
      } catch (error) {
        return new Response(error.message, { status: 404 });
      }
    });
    registerIpc();
    createWindow();
  });
}

app.on("before-quit", () => {
  terminateProcessTree(activeInference);
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
