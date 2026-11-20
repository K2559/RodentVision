# RodentVision

This branch of RodentVision is a Windows desktop app for reviewing littering, bird feeding, and dog occurrence in recorded video. It sends selected clips to Gemini Decision 1 for analysis and generates a Littering Act Report.

[Download page](https://k2559.github.io/RodentVision/) · [Windows releases](https://github.com/K2559/RodentVision/releases)

## Build from source

Use Node.js 22.12 or later in the 22.x series. From the repository root:

```sh
npm ci
npm run build
npm start
```

Run `npm run dev` for local development or `npm run dist` on Windows to package a portable executable. The React interface lives in `src/`, the desktop shell in `electron/`, and the Python inference runner and its tests in `backend/`.

```sh
python -m unittest discover -s backend -p "test_*.py"
```

## Analyze and report on littering events

1. Drop one or more video clips anywhere in the desktop app or select **Choose videos**.
2. Enter an API key with access to Gemini Decision 1. You can also set `GEMINI_API_KEY` before launching the app. The app keeps a key entered in the window in memory for that session.
3. Select a clip and click **Analyze with Gemini** to send that video to Google. Analyze each clip you want to include. The app uses one decision request with high-resolution video and supports clips up to 7 minutes 30 seconds.
4. Select a candidate timestamp to seek to it. Confirm or reject each candidate, add missed events at the current playback time, and mark the full clip reviewed when finished.
5. Open **Report** to save a CSV or PDF with one row per clip. The table shows confirmed timestamps, marks unfinished clips **REVIEW PENDING**, and uses **NA** for reviewed clips without confirmed events.

## Original V21 rodent workflow

Set `RODENTVISION_MODE=rodent` before launching from this checkout to open the original workflow.

1. Extract the Windows x64 ZIP and open `RodentVision/RodentVision.exe`.
2. Select a supported V21 `.pth` checkpoint and drop a video into **Source video**.
3. Choose an export folder, then select **Run inference**. Enable **Render review video** to export an annotated MP4 alongside the JSON detections.

## Inference requirements

Inference requires a separate compatible V21 model project and checkpoint, plus a Python environment with PyTorch, torchvision, and OpenCV. Review video export also requires FFmpeg.

Set `RODENTVISION_PROJECT_ROOT` to the V21 project folder and `RODENTVISION_PYTHON` to its Python executable before launching the app. Set `RODENTVISION_FFMPEG` to the FFmpeg executable if FFmpeg is outside your `PATH`.

The detector samples one frame every 10 seconds. The review export cadence is configured separately in **Advanced**.
