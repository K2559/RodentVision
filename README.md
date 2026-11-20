# RodentVision

RodentVision is a Windows desktop app for running V21 rodent detection on recorded thermal video and reviewing positive frames.

[Download page](https://k2559.github.io/RodentVision/) · [Windows releases](https://github.com/K2559/RodentVision/releases)

## Run a video

1. Extract the Windows x64 ZIP and open `RodentVision/RodentVision.exe`.
2. Select a supported V21 `.pth` checkpoint and drop a video into **Source video**.
3. Choose an export folder, then select **Run inference**. Enable **Render review video** to export an annotated MP4 alongside the JSON detections.

## Inference requirements

The download contains the desktop app. Inference requires the compatible V21 model project and checkpoint, plus a Python environment with PyTorch, torchvision, and OpenCV. Review video export also requires FFmpeg.

Set `RODENTVISION_PROJECT_ROOT` to the V21 project folder and `RODENTVISION_PYTHON` to its Python executable before launching the app. Set `RODENTVISION_FFMPEG` to the FFmpeg executable if FFmpeg is outside your `PATH`.
