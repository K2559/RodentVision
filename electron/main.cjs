const {
  app,
  BrowserWindow,
  dialog,
  ipcMain,
  protocol,
  shell,
} = require("electron");
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const { Readable } = require("node:stream");
const gemini = require("./gemini.cjs");

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
let activeGemini = null;

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
  ) || process.env.RODENTVISION_PROJECT_ROOT || KNOWN_PROJECT_ROOT;
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
  ipcMain.handle("gemini:configured", () => Boolean(process.env.GEMINI_API_KEY?.trim()));
  ipcMain.handle("gemini:analyze", async (event, options) => {
    if (activeGemini) throw new Error("Gemini is already analyzing a video.");
    const video = requirePath(options?.video, "Source video");
    const key = process.env.GEMINI_API_KEY?.trim() || options?.key?.trim();
    if (!key) throw new Error("Enter a Gemini API key.");
    const controller = new AbortController();
    activeGemini = controller;
    try {
      return await gemini.analyze({
        video,
        extension: path.extname(video).toLowerCase(),
        duration: options.duration,
        model: options.model,
        resolution: options.resolution,
        key,
        signal: controller.signal,
        onPhase: message => {
          if (!event.sender.isDestroyed()) event.sender.send("gemini:phase", message);
        },
      });
    } catch (error) {
      if (controller.signal.aborted) return null;
      throw error;
    } finally {
      activeGemini = null;
    }
  });
  ipcMain.handle("gemini:cancel", () => { activeGemini?.abort(); });
  ipcMain.handle("littering:save-report", async (event, format, csv) => {
    if (!["csv", "pdf"].includes(format)) throw new Error("Unsupported report format.");
    const result = await dialog.showSaveDialog(mainWindow, {
      title: "Save Littering Act Report",
      defaultPath: "Littering Act Report." + format,
      filters: [{ name: format.toUpperCase(), extensions: [format] }],
    });
    if (result.canceled || !result.filePath) return null;
    let contents;
    if (format === "pdf") {
      contents = await event.sender.printToPDF({ pageSize: "A4", printBackground: true, preferCSSPageSize: true });
    } else {
      if (typeof csv !== "string" || csv.length > 2_000_000) throw new Error("Invalid report content.");
      contents = "\uFEFF" + csv;
    }
    await fs.promises.writeFile(result.filePath, contents);
    return result.filePath;
  });
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
    title: process.env.RODENTVISION_MODE === "rodent" ? "RodentVision" : "RodentVision · Littering Event Review",
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

  const query = process.env.RODENTVISION_MODE === "rodent" ? { mode: "rodent" } : {};
  if (process.env.VITE_DEV_SERVER_URL) {
    const url = new URL(process.env.VITE_DEV_SERVER_URL);
    if (query.mode) url.searchParams.set("mode", query.mode);
    mainWindow.loadURL(url.toString());
  } else {
    mainWindow.loadFile(path.join(SOURCE_ROOT, "dist", "index.html"), { query });
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

if (process.env.RODENTVISION_MODE !== "rodent" && !app.commandLine.hasSwitch("user-data-dir")) {
  const profile = path.join(app.getPath("appData"), "RodentVision Littering");
  fs.mkdirSync(profile, { recursive: true });
  app.setPath("userData", profile);
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
        const { size } = fs.statSync(resolved);
        const type = {
          ".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime",
          ".mkv": "video/x-matroska", ".avi": "video/x-msvideo", ".webm": "video/webm",
        }[path.extname(resolved).toLowerCase()];
        const headers = { "Accept-Ranges": "bytes", "Content-Type": type };
        const range = request.headers?.get("range");
        let start = 0;
        let end = size - 1;
        if (range) {
          const match = /^bytes=(\d*)-(\d*)$/.exec(range);
          if (match && (match[1] || match[2])) {
            if (match[1]) {
              start = Number(match[1]);
              if (match[2]) end = Math.min(end, Number(match[2]));
            } else {
              start = Math.max(0, size - Number(match[2]));
            }
          }
          if (!match || (!match[1] && !match[2]) || start > end) {
            headers["Content-Range"] = "bytes */" + size;
            return new Response(null, { status: 416, headers });
          }
          headers["Content-Range"] = "bytes " + start + "-" + end + "/" + size;
        }
        headers["Content-Length"] = String(end - start + 1);
        const body = request.method === "HEAD" || size === 0
          ? null : Readable.toWeb(fs.createReadStream(resolved, { start, end }));
        return new Response(body, { status: range ? 206 : 200, headers });
      } catch (error) {
        return new Response(error.message, { status: 404 });
      }
    });
    registerIpc();
    createWindow();
  });
}

app.on("before-quit", () => {
  activeGemini?.abort();
  terminateProcessTree(activeInference);
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
