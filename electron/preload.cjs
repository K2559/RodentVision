const { contextBridge, ipcRenderer, webUtils } = require("electron");

contextBridge.exposeInMainWorld("rodentVision", {
  geminiConfigured: () => ipcRenderer.invoke("gemini:configured"),
  analyzeWithGemini: options => ipcRenderer.invoke("gemini:analyze", options),
  cancelGemini: () => ipcRenderer.invoke("gemini:cancel"),
  saveLitteringReport: (format, csv) => ipcRenderer.invoke("littering:save-report", format, csv),
  onGeminiPhase: callback => {
    const listener = (_event, message) => callback(message);
    ipcRenderer.on("gemini:phase", listener);
    return () => ipcRenderer.removeListener("gemini:phase", listener);
  },
  getEnvironment: () => ipcRenderer.invoke("environment:get"),
  chooseCheckpoint: () => ipcRenderer.invoke("dialog:checkpoint"),
  chooseVideo: () => ipcRenderer.invoke("dialog:video"),
  chooseVideos: () => ipcRenderer.invoke("dialog:videos"),
  chooseResult: () => ipcRenderer.invoke("dialog:result"),
  chooseOutputDirectory: () => ipcRenderer.invoke("dialog:output-directory"),
  inspectCheckpoint: (checkpoint) => ipcRenderer.invoke("checkpoint:inspect", checkpoint),
  startInference: (options) => ipcRenderer.invoke("inference:start", options),
  startBatchInference: (options) => ipcRenderer.invoke("inference:start-batch", options),
  cancelInference: () => ipcRenderer.invoke("inference:cancel"),
  loadResult: (resultPath) => ipcRenderer.invoke("result:load", resultPath),
  showItemInFolder: (targetPath) => ipcRenderer.invoke("shell:show-item", targetPath),
  openPath: (targetPath) => ipcRenderer.invoke("shell:open-path", targetPath),
  getPathForFile: (file) => webUtils.getPathForFile(file),
  mediaUrl: (targetPath) =>
    `rodent-media://local/video?path=${encodeURIComponent(targetPath)}`,
  onInferenceEvent: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on("inference:event", listener);
    return () => ipcRenderer.removeListener("inference:event", listener);
  },
});
