from __future__ import annotations

import argparse
from bisect import bisect_left
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RESULT_SCHEMA = "rodent-vision-inference/1.0"
DETECTION_FPS = 0.1
SUPPORTED_SCHEMAS = {
    "v21-pilot18-two-stage-source-checkpoint/1.1",
    "v21-pilot21-static-fcos-iou-quality-formal-checkpoint/1.1",
    "v21-pilot21-screened-quota-comparison-checkpoint/1.0",
    "v21-pilot22-manual-review-finetune-checkpoint/1.0",
}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
_cancel_requested = False


class CancelledError(RuntimeError):
    pass


def emit(event: str, **values: Any) -> None:
    print(json.dumps({"event": event, **values}, ensure_ascii=True), flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def finite_threshold(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and 0.0 <= parsed <= 1.0 else None


def find_project_root(checkpoint: Path) -> Path:
    candidates = [checkpoint.parent, *checkpoint.parents]
    configured = os.environ.get("RODENTVISION_PROJECT_ROOT")
    if configured:
        candidates.insert(0, Path(configured))
    for candidate in candidates:
        if (
            (candidate / "training_scripts" / "pilot18_two_stage_v21.py").is_file()
            and (candidate / "training_scripts" / "pilot21_fcos_iou_quality_v21.py").is_file()
        ):
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate the v21 model project from this checkpoint. "
        "Set RODENTVISION_PROJECT_ROOT to the project directory."
    )


def load_payload(checkpoint: Path) -> Mapping[str, Any]:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload is not a mapping")
    schema = payload.get("schema")
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError(f"unsupported checkpoint schema: {schema!r}")
    return payload


def model_config_for(
    payload: Mapping[str, Any], schema: str, project_root: Path
) -> Mapping[str, Any]:
    if schema == "v21-pilot21-static-fcos-iou-quality-formal-checkpoint/1.1":
        path = project_root / "configs" / "pilot21_fcos_iou_quality_source_bf16.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        return config["model"]
    if schema == "v21-pilot21-screened-quota-comparison-checkpoint/1.0":
        path = project_root / "configs" / "pilot21_screened_quota_comparison.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        return config["model"]
    config_payload = payload.get("config_payload")
    if not isinstance(config_payload, Mapping) or not isinstance(
        config_payload.get("model"), Mapping
    ):
        raise ValueError("checkpoint does not contain its model configuration")
    return config_payload["model"]


def checkpoint_model_info(
    checkpoint: Path,
    payload: Mapping[str, Any],
    *,
    include_sha256: bool = True,
) -> dict[str, Any]:
    import torch

    schema = str(payload["schema"])
    project_root = find_project_root(checkpoint)
    model_config = model_config_for(payload, schema, project_root)
    validation = payload.get("validation")
    if not isinstance(validation, Mapping):
        validation = {}

    presets: list[dict[str, Any]] = []

    def add_preset(label: str, value: Any, detail: str) -> None:
        threshold = finite_threshold(value)
        if threshold is None or any(
            math.isclose(threshold, item["value"], abs_tol=1e-9) for item in presets
        ):
            return
        presets.append({"label": label, "value": threshold, "detail": detail})

    if schema == "v21-pilot21-static-fcos-iou-quality-formal-checkpoint/1.1":
        family = "fcos"
        model_id = "pilot21-quality"
        display_name = "Pilot21 IoU Quality"
        architecture = "FCOS ResNet50 FPN P2-P7"
        epoch = int(payload.get("epoch", -1))
        add_preset(
            "Primary calibration",
            nested(validation, "selection", "primary", "threshold"),
            "Precision-gated source calibration",
        )
        add_preset(
            "Balanced review",
            nested(validation, "selection", "balanced_review", "threshold"),
            "Balanced calibration review point",
        )
        add_preset(
            "Recall review",
            nested(validation, "selection", "recall_oriented_review", "threshold"),
            "Recall-oriented calibration review point",
        )
    elif schema == "v21-pilot21-screened-quota-comparison-checkpoint/1.0":
        family = "fcos"
        arm = str(payload.get("arm", "control"))
        model_id = f"pilot21-screened-{arm}"
        display_name = f"Pilot21 Screened {arm.title()}"
        architecture = "FCOS ResNet50 FPN P2-P7"
        epoch = int(payload.get("epoch", -1))
        if arm == "control" and epoch == 2:
            add_preset(
                "Demo operating point",
                0.37109375,
                "Frozen external-demo operating point",
            )
        add_preset(
            "Primary calibration",
            nested(validation, "sweep", "selection", "primary", "threshold"),
            "Precision-gated combined validation",
        )
        add_preset(
            "Balanced review",
            nested(validation, "sweep", "selection", "balanced_review", "threshold"),
            "Balanced combined-validation review point",
        )
        add_preset(
            "Recall review",
            nested(validation, "sweep", "selection", "recall_oriented_review", "threshold"),
            "Recall-oriented combined-validation review point",
        )
    elif schema == "v21-pilot22-manual-review-finetune-checkpoint/1.0":
        family = "fasterrcnn"
        model_id = "pilot25-screened-quota"
        display_name = "Pilot25 Screened Quota"
        architecture = "Faster R-CNN ResNet50 FPN v2"
        epoch = int(payload.get("global_epoch", -1))
        add_preset("Demo review", 0.9, "Established demo review threshold")
        add_preset(
            "Source calibration",
            nested(validation, "source", "own_selected", "threshold"),
            "Source-only validation",
        )
        add_preset(
            "Combined calibration",
            nested(validation, "combined", "selected", "threshold"),
            "Source and manual validation",
        )
    else:
        family = "fasterrcnn"
        model_id = "pilot18-source"
        display_name = "Pilot18 Source"
        architecture = "Faster R-CNN ResNet50 FPN v2"
        epoch = int(payload.get("epoch", -1))
        add_preset(
            "Source calibration",
            nested(validation, "selected", "threshold"),
            "Precision-gated source calibration",
        )

    if not presets:
        add_preset("Review", 0.9, "General high-confidence review threshold")
    arm = str(payload.get("arm") or "detector")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpointSha256": sha256_file(checkpoint) if include_sha256 else None,
        "checkpointBytes": checkpoint.stat().st_size,
        "schema": schema,
        "family": family,
        "modelId": model_id,
        "displayName": display_name,
        "architecture": architecture,
        "architectureContract": model_config.get("architecture"),
        "epoch": epoch,
        "arm": arm,
        "optimizerUpdates": payload.get(
            "optimizer_updates", payload.get("cumulative_optimizer_updates")
        ),
        "parameterSha256": payload.get("model_parameter_sha256"),
        "createdAtUtc": payload.get("created_at_utc"),
        "defaultThreshold": presets[0]["value"],
        "thresholdPresets": presets,
        "input": {
            "minimumSize": model_config.get(
                "input_min_size", model_config.get("input_height")
            ),
            "maximumSize": model_config.get(
                "input_max_size", model_config.get("input_width")
            ),
        },
        "runtime": {
            "torch": torch.__version__,
            "cudaAvailable": torch.cuda.is_available(),
            "cudaVersion": torch.version.cuda,
            "gpu": gpu_name,
        },
        "projectRoot": str(project_root),
    }


def load_model(
    checkpoint: Path, device: Any
) -> tuple[Any, dict[str, Any], Mapping[str, Any]]:
    import torch

    emit("phase", phase="checkpoint", message="Reading checkpoint")
    payload = load_payload(checkpoint)
    info = checkpoint_model_info(checkpoint, payload, include_sha256=True)
    project_root = Path(info["projectRoot"])
    scripts = project_root / "training_scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    schema = info["schema"]
    model_config = model_config_for(payload, schema, project_root)

    emit("phase", phase="model", message=f"Building {info['displayName']}")
    if info["family"] == "fcos":
        from pilot21_fcos_iou_quality_v21 import build_pilot21_model, model_contract

        contract = payload.get("model_contract")
        quality_enabled = bool(nested(contract or {}, "quality_head", "enabled"))
        arm_config = model_config.get("candidate" if quality_enabled else "control", {})
        model = build_pilot21_model(
            quality_enabled=quality_enabled,
            backbone_name="resnet50",
            pretrained_backbone=False,
            trainable_backbone_layers=int(model_config["trainable_backbone_layers"]),
            input_height=int(model_config["input_height"]),
            input_width=int(model_config["input_width"]),
            detections_per_image=int(model_config["detections_per_image"]),
            score_threshold=float(model_config["inference_score_floor"]),
            nms_threshold=float(model_config["nms_threshold"]),
            quality_loss_weight=float(arm_config.get("quality_loss_weight", 1.0)),
        )
        actual_contract = model_contract(model)
    else:
        from pilot18_two_stage_v21 import build_pilot18_model, model_contract

        model = build_pilot18_model(model_config, pretrained_detector=False)
        actual_contract = model_contract(model)

    expected_contract = payload.get("model_contract")
    if expected_contract is not None and actual_contract != expected_contract:
        raise ValueError("checkpoint model contract does not match the reconstructed model")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint model state is missing")
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"checkpoint state mismatch: {result}")
    del state
    model.to(device).eval()
    return model, info, payload


def resolve_ffmpeg() -> Path:
    configured = os.environ.get("RODENTVISION_FFMPEG")
    candidates = [
        Path(configured) if configured else None,
        Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
    ]
    found = shutil.which("ffmpeg")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("ffmpeg.exe was not found")


def video_metadata(video: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    metadata = {
        "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frames": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
    }
    capture.release()
    if (
        metadata["width"] <= 0
        or metadata["height"] <= 0
        or metadata["fps"] <= 0
        or metadata["frames"] <= 0
    ):
        raise RuntimeError(f"invalid video metadata: {metadata}")
    metadata["durationSeconds"] = metadata["frames"] / metadata["fps"]
    return metadata


def sampling_frame_indexes(
    metadata: Mapping[str, Any], output_fps: float, max_frames: int | None = None
) -> list[int]:
    """Select the frame at each cadence boundary.

    The exported review video places frame ``n`` at ``n / output_fps`` seconds.
    Selecting the source frame at that same boundary keeps the JSON timestamp,
    the rendered frame, and the playback position referring to the same image.
    """
    source_frames = int(metadata["frames"])
    source_fps = float(metadata["fps"])
    if source_frames <= 0 or source_fps <= 0 or output_fps <= 0:
        raise ValueError("video and sampling rates must be positive")

    if output_fps >= source_fps or math.isclose(output_fps, source_fps):
        indexes = list(range(source_frames))
    else:
        duration = source_frames / source_fps
        interval = 1.0 / output_fps
        window_count = max(1, int(math.ceil(duration / interval - 1e-9)))
        indexes = []
        for window_index in range(window_count):
            window_start = window_index * interval
            # Use the first source frame in each cadence window.  ``round``
            # chooses the nearest representable source frame for fractional
            # source rates (for example 29.97 FPS) while remaining exact for
            # the integer rates used by the supported camera exports.
            frame_index = min(
                source_frames - 1,
                max(0, int(round(window_start * source_fps))),
            )
            if not indexes or indexes[-1] != frame_index:
                indexes.append(frame_index)

    if max_frames is not None:
        return indexes[: max(0, int(max_frames))]
    return indexes


def review_render_plan(
    metadata: Mapping[str, Any], output_fps: float, sample_count: int
) -> tuple[float, list[int], list[float]]:
    """Describe the playback timeline for boundary-aligned exported frames."""
    if output_fps <= 0 or sample_count <= 0:
        raise ValueError("sampling rate and sample count must be positive")

    if float(metadata["fps"]) <= 0:
        raise ValueError("source frame rate must be positive")

    timestamps = [sample_index / output_fps for sample_index in range(sample_count)]
    return output_fps, [1] * sample_count, timestamps


def render_positions_for_detection_indexes(
    detection_indexes: Sequence[int], render_indexes: Sequence[int]
) -> list[int]:
    """Map each sparse detection frame to its nearest exported frame.

    Detection and export cadence are intentionally independent.  When boxes
    are requested, they are applied only to the exported frame that represents
    the corresponding ten-second detection window; clean exports still contain
    every selected source frame.
    """
    if not render_indexes:
        raise ValueError("at least one rendered frame is required")
    positions: list[int] = []
    for detection_index in detection_indexes:
        insertion = bisect_left(render_indexes, detection_index)
        candidates = []
        if insertion > 0:
            candidates.append(insertion - 1)
        if insertion < len(render_indexes):
            candidates.append(insertion)
        position = min(
            candidates,
            key=lambda candidate: (
                abs(render_indexes[candidate] - detection_index),
                candidate,
            ),
        )
        positions.append(position)
    return positions


def filter_detections(
    output: Mapping[str, Any],
    *,
    family: str,
    threshold: float,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    import torch

    boxes = torch.as_tensor(output.get("boxes", []), dtype=torch.float32).detach().cpu()
    scores = torch.as_tensor(output.get("scores", []), dtype=torch.float32).detach().cpu()
    labels = torch.as_tensor(output.get("labels", []), dtype=torch.int64).detach().cpu()
    if boxes.ndim != 2 or boxes.shape[-1:] != (4,):
        raise ValueError(f"model returned invalid box shape {tuple(boxes.shape)}")
    if scores.ndim != 1 or labels.ndim != 1 or len(scores) != len(boxes) or len(labels) != len(boxes):
        raise ValueError("model returned misaligned boxes, scores, and labels")
    foreground_label = 0 if family == "fcos" else 1
    detections: list[dict[str, Any]] = []
    for raw_box, raw_score, raw_label in zip(
        boxes.tolist(), scores.tolist(), labels.tolist()
    ):
        score = float(raw_score)
        if int(raw_label) != foreground_label or not math.isfinite(score) or score < threshold:
            continue
        if len(raw_box) != 4 or not all(math.isfinite(float(value)) for value in raw_box):
            continue
        x1 = max(0.0, min(float(width), float(raw_box[0])))
        y1 = max(0.0, min(float(height), float(raw_box[1])))
        x2 = max(0.0, min(float(width), float(raw_box[2])))
        y2 = max(0.0, min(float(height), float(raw_box[3])))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            {
                "classId": 1,
                "className": "rodent",
                "score": score,
                "bboxXyxyPixels": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
                "bboxXyxyNormalized": [
                    round(x1 / width, 7),
                    round(y1 / height, 7),
                    round(x2 / width, 7),
                    round(y2 / height, 7),
                ],
                "areaPixels": round((x2 - x1) * (y2 - y1), 3),
            }
        )
    detections.sort(key=lambda item: (-item["score"], item["bboxXyxyPixels"]))
    return detections


def predict(
    model: Any,
    frames: Sequence[Any],
    *,
    device: Any,
    family: str,
    threshold: float,
) -> list[list[dict[str, Any]]]:
    import torch

    tensors = []
    for frame in frames:
        tensor = torch.from_numpy(frame).to(device, non_blocking=True)
        tensor = tensor[..., [2, 1, 0]].permute(2, 0, 1).contiguous().float().div_(255.0)
        tensors.append(tensor)
    if device.type == "cuda":
        dtype = torch.bfloat16 if family == "fcos" and torch.cuda.is_bf16_supported() else torch.float16
        autocast = torch.autocast(device_type="cuda", dtype=dtype)
    else:
        autocast = nullcontext()
    with torch.inference_mode(), autocast:
        outputs = model(tensors)
    if len(outputs) != len(frames):
        raise RuntimeError("model output count does not match the input batch")
    height, width = frames[0].shape[:2]
    return [
        filter_detections(
            output,
            family=family,
            threshold=threshold,
            width=width,
            height=height,
        )
        for output in outputs
    ]


def draw_boxes(frame: Any, detections: Sequence[Mapping[str, Any]]) -> None:
    """Draw detection rectangles without text so audit tools can add labels."""
    import cv2

    height, width = frame.shape[:2]
    thickness = max(1, int(round(min(width, height) / 480)))
    color = (48, 214, 177)
    for detection in detections:
        raw_x1, raw_y1, raw_x2, raw_y2 = [
            int(round(value)) for value in detection["bboxXyxyPixels"]
        ]
        x1 = min(width - 1, max(0, min(raw_x1, raw_x2)))
        y1 = min(height - 1, max(0, min(raw_y1, raw_y2)))
        x2 = min(width - 1, max(0, max(raw_x1, raw_x2)))
        y2 = min(height - 1, max(0, max(raw_y1, raw_y2)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def render_dimensions(width: int, height: int, resolution: str) -> tuple[int, int]:
    """Return even output dimensions without upscaling a source smaller than 480p."""
    if width <= 0 or height <= 0:
        raise ValueError("video dimensions must be positive")
    if resolution not in {"source", "480p"}:
        raise ValueError("render resolution must be source or 480p")
    target_width = int(width)
    target_height = int(height)
    if resolution == "480p" and height > 480:
        target_height = 480
        target_width = max(2, int(round(width * target_height / height)))
    # yuv420p requires even dimensions. Preserve the aspect ratio as closely
    # as possible while avoiding an expensive padding pass in the encoder.
    target_width += target_width % 2
    target_height += target_height % 2
    return target_width, target_height


class VideoEncoder:
    def __init__(
        self,
        ffmpeg: Path,
        output: Path,
        *,
        width: int,
        height: int,
        input_fps: float,
        output_fps: float,
        duration_seconds: float,
        prefer_nvenc: bool,
    ) -> None:
        self.output = output
        self.output.parent.mkdir(parents=True, exist_ok=True)
        # Keep the growing file outside the export directory.  External upload
        # watchers can safely scan the target folder while FFmpeg is encoding;
        # only the completed file is moved there by ``finish``.
        # Prefer the system temporary volume when it shares a filesystem with
        # the export folder.  If exports are on another drive, use a hidden
        # sibling directory on that drive so the final ``replace`` stays
        # atomic.  Either location remains outside the watched export folder.
        output_parent = self.output.parent.resolve()
        system_temporary_parent = Path(tempfile.gettempdir()).resolve()
        try:
            same_filesystem = os.stat(output_parent).st_dev == os.stat(
                system_temporary_parent
            ).st_dev
        except OSError:
            same_filesystem = False
        staging_parent = (
            system_temporary_parent
            if same_filesystem
            else output_parent.parent
        )
        self._temporary_directory: Path | None = Path(
            tempfile.mkdtemp(
                prefix=".rodentvision-render-",
                dir=str(staging_parent),
            )
        )
        self.partial = self._temporary_directory / f".{output.stem}.partial{output.suffix}"
        self.width = width + (width % 2)
        self.height = height + (height % 2)
        self.input_fps = float(input_fps)
        self.fps = float(output_fps)
        self.duration_seconds = max(0.0, float(duration_seconds))
        self.input_frames_written = 0
        self.partial.unlink(missing_ok=True)
        encoder = "h264_nvenc" if prefer_nvenc else "libx264"
        command = [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.input_fps:.8f}",
            "-i",
            "-",
            "-an",
        ]
        # Input and output rates normally match. The optional conversion remains
        # available for callers that explicitly request a different playback rate.
        if not math.isclose(self.input_fps, self.fps, rel_tol=1e-6, abs_tol=1e-6):
            command += ["-vf", f"fps={self.fps:.8f}"]
        if self.duration_seconds > 0:
            command += ["-t", f"{self.duration_seconds:.6f}"]
        command += ["-c:v", encoder]
        if encoder == "h264_nvenc":
            # This FFmpeg build exposes the NVENC preset as an integer enum.
            command += ["-preset", "4", "-rc", "vbr", "-cq", "22", "-b:v", "0"]
        else:
            command += ["-preset", "medium", "-crf", "20"]
        command += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.partial)]
        self.encoder_name = encoder
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except Exception:
            # ``mkdtemp`` succeeds before FFmpeg is launched.  Do not leave an
            # orphaned staging directory behind when process creation fails.
            self._cleanup_temporary_directory()
            raise
        if self.process.stdin is None or self.process.stderr is None:
            self.abort()
            raise RuntimeError("failed to start FFmpeg video encoder")

    def write(self, frame: Any) -> None:
        import cv2

        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.copyMakeBorder(
                frame,
                0,
                self.height - frame.shape[0],
                0,
                self.width - frame.shape[1],
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        try:
            self.process.stdin.write(frame.tobytes())
            self.input_frames_written += 1
        except (BrokenPipeError, OSError) as error:
            detail = self.process.stderr.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"FFmpeg encoder stopped: {detail or error}") from error

    def finish(self) -> dict[str, Any]:
        try:
            self.process.stdin.close()
            code = self.process.wait()
            detail = self.process.stderr.read().decode("utf-8", errors="replace").strip()
            if code != 0:
                raise RuntimeError(f"FFmpeg encoder failed: {detail or f'exit code {code}'}")
            # ``replace`` is atomic on the same filesystem and is the first
            # operation that makes a completed video visible in the export
            # directory.
            self.partial.replace(self.output)
            frames = max(
                1,
                int(round(self.duration_seconds * self.fps))
                if self.duration_seconds > 0
                else self.input_frames_written,
            )
            duration = (
                self.duration_seconds
                if self.duration_seconds > 0
                else frames / max(self.fps, 1e-9)
            )
            return {
                "path": str(self.output),
                "bytes": self.output.stat().st_size,
                "width": self.width,
                "height": self.height,
                "encoder": self.encoder_name,
                "fps": self.fps,
                "frames": frames,
                "inputFrames": self.input_frames_written,
                "durationSeconds": duration,
            }
        finally:
            self._cleanup_temporary_directory()

    def _cleanup_temporary_directory(self) -> None:
        temporary_directory = self._temporary_directory
        if temporary_directory is not None:
            shutil.rmtree(temporary_directory, ignore_errors=True)
            self._temporary_directory = None

    def abort(self) -> None:
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            try:
                process.stdin.close()
            except Exception:
                pass
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        partial = getattr(self, "partial", None)
        if partial is not None:
            partial.unlink(missing_ok=True)
        self._cleanup_temporary_directory()


def ffmpeg_has_nvenc(ffmpeg: Path) -> bool:
    listed = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode != 0 or "h264_nvenc" not in listed.stdout:
        return False
    # Encoder listings can include NVENC even when the installed driver cannot
    # open a session. Probe one tiny frame so rendering falls back to libx264.
    probe = subprocess.run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:r=1:d=0.1",
            "-frames:v",
            "1",
            "-an",
            "-c:v",
            "h264_nvenc",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


def safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned[:64] or "video"


def unique_output_paths(
    output_dir: Path, video: Path, model_id: str, threshold: float, render: bool
) -> tuple[Path, Path | None]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    threshold_slug = f"{threshold:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    base = f"{safe_slug(video.stem)}_{safe_slug(model_id)}_t{threshold_slug}_{stamp}"
    candidate = base
    suffix = 2
    while (output_dir / f"{candidate}.json").exists() or (
        render and (output_dir / f"{candidate}.mp4").exists()
    ):
        candidate = f"{base}_{suffix}"
        suffix += 1
    return output_dir / f"{candidate}.json", (
        output_dir / f"{candidate}.mp4" if render else None
    )


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=False, ensure_ascii=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def copy_original_video(
    video: Path,
    output_dir: Path,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    source = video.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    source_bytes = source.stat().st_size
    direct_destination = output_dir / source.name
    if direct_destination == source:
        if progress:
            progress(source_bytes, source_bytes)
        return source

    destination = direct_destination
    if destination.exists():
        base = f"{source.stem}_original"
        destination = output_dir / f"{base}{source.suffix}"
        suffix = 2
        while destination.exists():
            destination = output_dir / f"{base}_{suffix}{source.suffix}"
            suffix += 1

    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.partial"
    )
    copied_bytes = 0
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            while True:
                if _cancel_requested:
                    raise CancelledError("Inference cancelled")
                chunk = reader.read(8 * 1024 * 1024)
                if not chunk:
                    break
                writer.write(chunk)
                copied_bytes += len(chunk)
                if progress:
                    progress(copied_bytes, source_bytes)
            writer.flush()
            os.fsync(writer.fileno())
        shutil.copystat(source, temporary)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def stage_original_videos(
    videos: Sequence[Path],
    output_dir: Path,
    progress: Callable[[int, int, int, int, Path], None] | None = None,
) -> list[dict[str, str]]:
    if not videos:
        raise ValueError("at least one original video is required")

    sources = [video.expanduser().resolve() for video in videos]
    for source in sources:
        if not source.is_file() or source.suffix.lower() not in VIDEO_EXTENSIONS:
            raise FileNotFoundError(f"unsupported or missing video: {source}")

    resolved_output_dir = output_dir.expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = sum(source.stat().st_size for source in sources)
    completed_bytes = 0
    staged: list[dict[str, str]] = []
    for index, source in enumerate(sources):
        source_bytes = source.stat().st_size

        def report_file_progress(copied_bytes: int, _source_bytes: int) -> None:
            if progress:
                progress(
                    index + 1,
                    len(sources),
                    completed_bytes + copied_bytes,
                    total_bytes,
                    source,
                )

        copied = copy_original_video(source, resolved_output_dir, report_file_progress)
        completed_bytes += source_bytes
        staged.append({"video": str(source), "originalVideo": str(copied)})

    return staged


def resolve_device(requested: str) -> Any:
    import torch

    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def automatic_batch_size(device: Any, family: str) -> int:
    """Choose a throughput-oriented batch from currently available VRAM.

    The old FCOS heuristic only selected four images when a GPU had at least
    20 GiB free. That can never happen on a 16 GiB RTX 4080 SUPER, so the
    packaged app silently fell back to two-image batches. The thresholds below
    are based on the measured peak for this model (about 8.9 GiB at 64
    images); ``predict_with_fallback`` still halves the batch if another GPU
    workload leaves less headroom than expected.
    """
    import torch

    if device.type != "cuda":
        return 1
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    free_gib = free_bytes / 1024**3
    if family == "fcos":
        if free_gib >= 12.0:
            return 64
        if free_gib >= 7.0:
            return 32
        if free_gib >= 4.0:
            return 16
        if free_gib >= 2.5:
            return 8
        if free_gib >= 1.5:
            return 4
    elif free_gib >= 10.0:
        return 16
    elif free_gib >= 6.0:
        return 8
    elif free_gib >= 3.0:
        return 4
    if free_gib >= 8.0:
        return 2
    return 1


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint.expanduser().resolve()
    video = args.video.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not video.is_file() or video.suffix.lower() not in VIDEO_EXTENSIONS:
        raise FileNotFoundError(f"unsupported or missing video: {video}")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    requested_output_fps = getattr(args, "output_fps", None)
    if requested_output_fps is None:
        requested_output_fps = getattr(args, "sample_fps", 1.0)
    requested_output_fps = float(requested_output_fps)
    if not math.isfinite(requested_output_fps) or requested_output_fps < 0.0:
        raise ValueError("output FPS must be a finite, non-negative number")
    if args.batch_size < 0:
        raise ValueError("batch size cannot be negative")
    render_annotations = bool(getattr(args, "render_annotations", False))
    render_resolution = getattr(args, "render_resolution", "source")
    copy_original_requested = bool(getattr(args, "copy_original", False))
    staged_original = getattr(args, "staged_original", None)
    if render_resolution not in {"source", "480p"}:
        raise ValueError("render resolution must be source or 480p")
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    copied_original_path = None
    if staged_original is not None:
        copied_original_path = staged_original.expanduser().resolve()
        if not copied_original_path.is_file():
            raise FileNotFoundError(f"missing staged original video: {copied_original_path}")
        if os.path.normcase(str(copied_original_path.parent)) != os.path.normcase(
            str(output_dir)
        ):
            raise ValueError("the staged original video must be inside the export folder")
    elif copy_original_requested:
        emit("phase", phase="copy", message="Copying original video before inference")
        copied_original_path = copy_original_video(video, output_dir)
    copy_original = copied_original_path is not None

    emit("phase", phase="libraries", message="Loading inference libraries")
    import cv2
    import torch

    emit("phase", phase="metadata", message="Reading video metadata")
    metadata = video_metadata(video)
    output_fps = metadata["fps"] if requested_output_fps == 0 else min(
        requested_output_fps, metadata["fps"]
    )
    detection_fps = min(DETECTION_FPS, metadata["fps"])
    detection_indexes = sampling_frame_indexes(
        metadata, detection_fps, args.max_frames
    )
    render_indexes = sampling_frame_indexes(
        metadata, output_fps, args.max_frames
    )
    estimated_frames = len(detection_indexes)
    render_estimated_frames = len(render_indexes)
    if estimated_frames <= 0:
        raise RuntimeError("no video frames were selected for inference")
    if render_estimated_frames <= 0:
        raise RuntimeError("no video frames were selected for rendering")

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)
    model, model_info, payload = load_model(checkpoint, device)
    del payload
    batch_size = (
        int(args.batch_size)
        if args.batch_size > 0
        else automatic_batch_size(device, model_info["family"])
    )
    batch_mode = "manual" if args.batch_size > 0 else "auto"
    emit(
        "model",
        model=model_info,
        device=str(device),
        batchSize=batch_size,
        batchMode=batch_mode,
    )
    emit(
        "phase",
        phase="batch",
        message=f"Using {batch_size}-frame GPU batches ({batch_mode})",
        totalFrames=estimated_frames,
    )

    result_path, rendered_path = unique_output_paths(
        output_dir, video, model_info["modelId"], args.threshold, args.render
    )
    ffmpeg = None
    render_fps = output_fps
    render_output_timestamps: list[float] = []
    detection_render_positions: list[int] = []
    if rendered_path is not None:
        ffmpeg = resolve_ffmpeg()
        render_fps, _render_frame_holds, render_output_timestamps = review_render_plan(
            metadata, output_fps, render_estimated_frames
        )
        detection_render_positions = render_positions_for_detection_indexes(
            detection_indexes, render_indexes
        )

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video}")

    emit(
        "phase",
        phase="inference",
        message="Running inference",
        totalFrames=estimated_frames,
    )
    frame_records: list[dict[str, Any]] = []
    batch_frames: list[Any] = []
    batch_metadata: list[dict[str, Any]] = []
    selected_count = 0
    processed_count = 0
    source_index = -1
    positive_frames = 0
    total_detections = 0
    maximum_score = 0.0
    score_sum = 0.0
    rendered_info = None
    render_width, render_height = render_dimensions(
        metadata["width"], metadata["height"], render_resolution
    )
    source_frames_read = 0
    render_detections_by_position: dict[int, list[dict[str, Any]]] = {}
    last_sampling_report = -1
    sampling_report_interval = max(1, int(metadata["frames"] * 0.05))
    # Do not use OpenCV's frame-seek path for sparse H.264 sampling on Windows.
    # The bundled opencv_videoio_ffmpeg backend can terminate Python with
    # STATUS_ILLEGAL_INSTRUCTION (0xC000001D) when seeking into long 1280x960
    # clips. Sequential decode is deterministic, and the discarded frames are
    # cheap compared with model inference.
    def predict_with_fallback(frames: Sequence[Any]) -> list[list[dict[str, Any]]]:
        """Retry a CUDA batch at half size when another process uses VRAM."""
        nonlocal batch_size
        try:
            return predict(
                model,
                frames,
                device=device,
                family=model_info["family"],
                threshold=float(args.threshold),
            )
        except RuntimeError as error:
            message = str(error).lower()
            if device.type != "cuda" or "out of memory" not in message or len(frames) <= 1:
                raise
            next_size = max(1, len(frames) // 2)
            if next_size >= batch_size:
                next_size = max(1, batch_size // 2)
            batch_size = next_size
            torch.cuda.empty_cache()
            emit(
                "phase",
                phase="batch",
                message=f"GPU memory is busy; reducing batch to {batch_size}",
                totalFrames=estimated_frames,
            )
            return predict_with_fallback(frames[:next_size]) + predict_with_fallback(
                frames[next_size:]
            )

    def flush_batch() -> None:
        nonlocal processed_count, positive_frames, total_detections, maximum_score, score_sum
        if not batch_frames:
            return
        if _cancel_requested:
            raise CancelledError("Inference cancelled")
        emit(
            "phase",
            phase="inference",
            message=f"Running inference ({processed_count}/{estimated_frames} frames)",
            totalFrames=estimated_frames,
        )
        predictions = predict_with_fallback(batch_frames)
        for frame_meta, detections in zip(batch_metadata, predictions):
            detection_scores = [float(item["score"]) for item in detections]
            positive_frames += int(bool(detections))
            total_detections += len(detections)
            if detection_scores:
                maximum_score = max(maximum_score, max(detection_scores))
                score_sum += sum(detection_scores)
            record = {
                **frame_meta,
                "detectionCount": len(detections),
                "maximumScore": max(detection_scores, default=None),
                "detections": detections,
            }
            frame_records.append(record)
            render_position = frame_meta.get("outputFrameIndex")
            if isinstance(render_position, int):
                render_detections_by_position.setdefault(render_position, []).extend(detections)
            processed_count += 1
        batch_frames.clear()
        batch_metadata.clear()
        elapsed = max(1e-6, time.monotonic() - started)
        rate = processed_count / elapsed
        remaining = max(0, estimated_frames - processed_count)
        emit(
            "progress",
            processedFrames=processed_count,
            totalFrames=estimated_frames,
            percent=min(100.0, processed_count * 100.0 / max(1, estimated_frames)),
            framesPerSecond=rate,
            etaSeconds=remaining / max(rate, 1e-6),
            detections=total_detections,
            positiveFrames=positive_frames,
        )

    try:
        for selected_count, target_index in enumerate(detection_indexes):
            if _cancel_requested:
                raise CancelledError("Inference cancelled")
            while source_index + 1 < target_index:
                if not capture.grab():
                    break
                source_index += 1
                source_frames_read = source_index + 1
                if source_index - last_sampling_report >= sampling_report_interval:
                    last_sampling_report = source_index
                    sampling_percent = min(
                        99.9,
                        source_frames_read * 100.0 / max(1, metadata["frames"]),
                    )
                    emit(
                        "phase",
                        phase="sampling",
                        message=f"Reading video samples ({sampling_percent:.1f}%)",
                        totalFrames=estimated_frames,
                    )
            ok, frame = capture.read()
            source_index += 1
            source_frames_read = source_index + 1
            if not ok or frame is None:
                break
            output_frame_index = (
                detection_render_positions[selected_count]
                if rendered_path is not None
                else None
            )
            frame_meta = {
                "sampleIndex": selected_count,
                "sourceFrameIndex": source_index,
                "timestampSeconds": source_index / metadata["fps"],
                "outputTimestampSeconds": (
                    render_output_timestamps[output_frame_index]
                    if output_frame_index is not None
                    else source_index / metadata["fps"]
                ),
            }
            if output_frame_index is not None:
                frame_meta["outputFrameIndex"] = output_frame_index
            batch_frames.append(frame)
            batch_metadata.append(frame_meta)
            if len(batch_frames) >= batch_size:
                flush_batch()
        flush_batch()
        if processed_count == 0:
            raise RuntimeError("no video frames were selected for inference")
    finally:
        capture.release()

    # Rendering is a second sequential decode pass. This keeps model work at
    # the fixed ten-second cadence while allowing the review video to contain
    # genuinely distinct one-second frames when that export option is chosen.
    if rendered_path is not None:
        encoder = VideoEncoder(
            ffmpeg,
            rendered_path,
            width=render_width,
            height=render_height,
            input_fps=render_fps,
            output_fps=render_fps,
            duration_seconds=0.0,
            prefer_nvenc=device.type == "cuda" and ffmpeg_has_nvenc(ffmpeg),
        )
        render_capture = cv2.VideoCapture(str(video))
        if not render_capture.isOpened():
            encoder.abort()
            raise RuntimeError(f"cannot open video for rendering: {video}")
        render_source_index = -1
        rendered_count = 0
        render_started = time.monotonic()
        render_report_interval = max(1, int(render_estimated_frames * 0.05))
        last_render_report = -1
        try:
            emit(
                "phase",
                phase="render",
                message=f"Rendering review video ({render_fps:g} FPS)",
                totalFrames=render_estimated_frames,
            )
            for output_frame_index, target_index in enumerate(render_indexes):
                if _cancel_requested:
                    raise CancelledError("Inference cancelled")
                while render_source_index + 1 < target_index:
                    if not render_capture.grab():
                        break
                    render_source_index += 1
                    source_frames_read = max(source_frames_read, render_source_index + 1)
                ok, frame = render_capture.read()
                render_source_index += 1
                source_frames_read = max(source_frames_read, render_source_index + 1)
                if not ok or frame is None:
                    break
                if render_annotations:
                    detections = render_detections_by_position.get(output_frame_index)
                    if detections:
                        draw_boxes(frame, detections)
                if frame.shape[1] != render_width or frame.shape[0] != render_height:
                    frame = cv2.resize(
                        frame,
                        (render_width, render_height),
                        interpolation=cv2.INTER_AREA,
                    )
                encoder.write(frame)
                rendered_count += 1
                if (
                    output_frame_index - last_render_report >= render_report_interval
                    or rendered_count == render_estimated_frames
                ):
                    last_render_report = output_frame_index
                    elapsed_render = max(1e-6, time.monotonic() - render_started)
                    render_rate = rendered_count / elapsed_render
                    remaining_render = max(0, render_estimated_frames - rendered_count)
                    emit(
                        "progress",
                        processedFrames=rendered_count,
                        totalFrames=render_estimated_frames,
                        percent=min(
                            100.0,
                            rendered_count * 100.0 / max(1, render_estimated_frames),
                        ),
                        framesPerSecond=render_rate,
                        etaSeconds=remaining_render / max(render_rate, 1e-6),
                        detections=total_detections,
                        positiveFrames=positive_frames,
                    )
            if rendered_count == 0:
                raise RuntimeError("no video frames were selected for rendering")
            emit("phase", phase="encoding", message="Finalizing review video")
            rendered_info = encoder.finish()
            encoder = None
        except Exception:
            encoder.abort()
            raise
        finally:
            render_capture.release()

    elapsed = time.monotonic() - started
    peak_gpu_gib = (
        torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
    )
    summary = {
        "processedFrames": processed_count,
        "positiveFrames": positive_frames,
        "negativeFrames": processed_count - positive_frames,
        "totalDetections": total_detections,
        "maximumScore": maximum_score if total_detections else None,
        "meanDetectionScore": score_sum / total_detections if total_detections else None,
        "elapsedSeconds": elapsed,
        "averageProcessedFps": processed_count / max(elapsed, 1e-9),
        "precision": None,
        "recall": None,
        "accuracyMetricsAvailable": False,
    }
    result = {
        "schema": RESULT_SCHEMA,
        "createdAtUtc": utc_now(),
        "status": "completed",
        "input": {
            "video": str(video),
            "videoBytes": video.stat().st_size,
            **metadata,
            "sourceFramesRead": source_frames_read,
        },
        "model": model_info,
        "settings": {
            "threshold": float(args.threshold),
            "detectionFps": detection_fps,
            "detectionIntervalSeconds": 1.0 / detection_fps,
            "outputFps": output_fps,
            "requestedOutputFps": requested_output_fps,
            "outputIntervalSeconds": 1.0 / output_fps,
            # Legacy field names remain as aliases for the model sampling
            # cadence, which is intentionally fixed at one frame per 10 sec.
            "sampleFps": detection_fps,
            "requestedSampleFps": detection_fps,
            "samplingIntervalSeconds": 1.0 / detection_fps,
            "renderPlaybackFps": render_fps if args.render else None,
            "renderAnnotations": bool(args.render and render_annotations),
            "renderResolution": render_resolution if args.render else None,
            "copyOriginalVideo": copy_original,
            "batchSize": batch_size,
            "batchMode": batch_mode,
            "samplingMode": "detection-window-boundary-sequential-read",
            "renderSamplingMode": "output-window-boundary-sequential-read",
            "device": str(device),
            "render": bool(args.render),
        },
        "outputs": {
            "json": str(result_path),
            "renderedVideo": str(rendered_path) if rendered_path else None,
            "originalVideo": str(copied_original_path) if copied_original_path else None,
            "renderedVideoInfo": rendered_info,
        },
        "summary": summary,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peakGpuMemoryGiB": peak_gpu_gib,
            "ffmpeg": str(ffmpeg) if ffmpeg else None,
            "encoder": rendered_info.get("encoder") if rendered_info else None,
        },
        "frames": frame_records,
        "metricNote": "Precision and recall require ground-truth labels and are not computed for unlabeled videos.",
    }
    emit("phase", phase="json", message="Saving JSON result")
    write_json_atomic(result_path, result)
    emit(
        "complete",
        resultPath=str(result_path),
        renderedVideo=str(rendered_path) if rendered_path else None,
        originalVideo=str(copied_original_path) if copied_original_path else None,
        summary=summary,
    )
    return result


def handle_signal(_signum: int, _frame: Any) -> None:
    global _cancel_requested
    _cancel_requested = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RodentVision inference backend")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a supported checkpoint")
    inspect_parser.add_argument("--checkpoint", type=Path, required=True)

    stage_parser = subparsers.add_parser(
        "stage-originals", help="Copy source videos into an export folder before inference"
    )
    stage_parser.add_argument("--video", dest="videos", type=Path, action="append", required=True)
    stage_parser.add_argument("--output-dir", type=Path, required=True)

    run_parser = subparsers.add_parser("run", help="Run video inference")
    run_parser.add_argument("--checkpoint", type=Path, required=True)
    run_parser.add_argument("--video", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--threshold", type=float, required=True)
    run_parser.add_argument("--output-fps", type=float, default=None)
    # Keep the old flag accepted for scripts and result reproducers. Its
    # meaning is now export cadence; model detection cadence is fixed.
    run_parser.add_argument("--sample-fps", type=float, default=1.0, help=argparse.SUPPRESS)
    run_parser.add_argument("--batch-size", type=int, default=0, help="zero selects automatically")
    run_parser.add_argument("--device", default="auto")
    run_parser.add_argument(
        "--render", action=argparse.BooleanOptionalAction, default=True
    )
    run_parser.add_argument(
        "--render-annotations", action=argparse.BooleanOptionalAction, default=False
    )
    run_parser.add_argument(
        "--render-resolution",
        choices=("source", "480p"),
        default="source",
        help="review video dimensions; detection always runs on the source resolution",
    )
    run_parser.add_argument(
        "--copy-original", action=argparse.BooleanOptionalAction, default=False
    )
    run_parser.add_argument("--staged-original", type=Path, default=None, help=argparse.SUPPRESS)
    run_parser.add_argument("--max-frames", type=int, default=None, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    args = build_parser().parse_args()
    try:
        if args.command == "inspect":
            checkpoint = args.checkpoint.expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            payload = load_payload(checkpoint)
            info = checkpoint_model_info(checkpoint, payload, include_sha256=True)
            emit("model_info", model=info)
        elif args.command == "stage-originals":
            emit(
                "phase",
                phase="copy",
                message=f"Copying all original videos (0/{len(args.videos)})",
            )

            def report_copy_progress(
                video_index: int,
                video_total: int,
                copied_bytes: int,
                total_bytes: int,
                video: Path,
            ) -> None:
                percent = copied_bytes * 100.0 / max(1, total_bytes)
                emit(
                    "copy_progress",
                    videoIndex=video_index,
                    videoTotal=video_total,
                    copiedBytes=copied_bytes,
                    totalBytes=total_bytes,
                    percent=min(100.0, percent),
                    video=str(video),
                )

            staged = stage_original_videos(
                args.videos,
                args.output_dir,
                progress=report_copy_progress,
            )
            emit("staged", originals=staged)
        else:
            run_inference(args)
        return 0
    except CancelledError as error:
        emit("cancelled", message=str(error))
        return 2
    except Exception as error:
        emit("error", message=str(error), errorType=type(error).__name__)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
