const { contextBridge, ipcRenderer, webUtils } = require("electron");

contextBridge.exposeInMainWorld("rodentVision", {
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
