import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from backend.inference_runner import (
    build_parser,
    copy_original_video,
    draw_boxes,
    render_positions_for_detection_indexes,
    render_dimensions,
    review_render_plan,
    sampling_frame_indexes,
    stage_original_videos,
    VideoEncoder,
)


class OutputOptionTests(unittest.TestCase):
    def test_cli_keeps_export_cadence_separate_from_detection_cadence(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "--checkpoint",
                "model.pth",
                "--video",
                "input.mp4",
                "--output-dir",
                "exports",
                "--threshold",
                "0.371",
                "--output-fps",
                "1",
                "--no-render-annotations",
                "--render-resolution",
                "480p",
                "--copy-original",
            ]
        )

        self.assertEqual(args.output_fps, 1.0)
        self.assertFalse(args.render_annotations)
        self.assertEqual(args.render_resolution, "480p")
        self.assertTrue(args.copy_original)

    def test_copy_original_video_preserves_bytes_and_avoids_overwrite(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = root / "source"
            output_dir = root / "exports"
            source_dir.mkdir()
            output_dir.mkdir()
            source = source_dir / "clip.mp4"
            source.write_bytes(b"original-video-bytes")
            existing = output_dir / source.name
            existing.write_bytes(b"existing-export")

            copied = copy_original_video(source, output_dir)

            self.assertEqual(existing.read_bytes(), b"existing-export")
            self.assertEqual(copied.name, "clip_original.mp4")
            self.assertEqual(copied.read_bytes(), source.read_bytes())

    def test_copy_original_video_reuses_source_already_in_export_folder(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            source = output_dir / "clip.mp4"
            source.write_bytes(b"original-video-bytes")

            copied = copy_original_video(source, output_dir)

            self.assertEqual(copied, source.resolve())
            self.assertEqual(list(output_dir.iterdir()), [source])

    def test_stage_original_videos_copies_the_entire_batch_in_source_order(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = root / "source"
            output_dir = root / "exports"
            source_dir.mkdir()
            first = source_dir / "first.mp4"
            second = source_dir / "second.mov"
            first.write_bytes(b"first-original")
            second.write_bytes(b"second-original")
            updates: list[tuple[int, int]] = []

            staged = stage_original_videos(
                [first, second],
                output_dir,
                progress=lambda index, total, *_values: updates.append((index, total)),
            )

            self.assertEqual(
                [item["video"] for item in staged],
                [str(first.resolve()), str(second.resolve())],
            )
            self.assertEqual((output_dir / "first.mp4").read_bytes(), first.read_bytes())
            self.assertEqual((output_dir / "second.mov").read_bytes(), second.read_bytes())
            self.assertIn((1, 2), updates)
            self.assertEqual(updates[-1], (2, 2))

    def test_stage_originals_cli_accepts_the_complete_video_batch(self) -> None:
        args = build_parser().parse_args(
            [
                "stage-originals",
                "--video",
                "first.mp4",
                "--video",
                "second.mp4",
                "--output-dir",
                "exports",
            ]
        )

        self.assertEqual(args.command, "stage-originals")
        self.assertEqual(args.videos, [Path("first.mp4"), Path("second.mp4")])

    def test_render_dimensions_scale_height_and_keep_even_pixels(self) -> None:
        self.assertEqual(render_dimensions(1280, 720, "480p"), (854, 480))
        self.assertEqual(render_dimensions(640, 480, "480p"), (640, 480))
        self.assertEqual(render_dimensions(1279, 719, "source"), (1280, 720))

    def test_old_sample_fps_flag_remains_an_export_alias(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "--checkpoint",
                "model.pth",
                "--video",
                "input.mp4",
                "--output-dir",
                "exports",
                "--threshold",
                "0.371",
                "--sample-fps",
                "0.1",
            ]
        )

        self.assertIsNone(args.output_fps)
        self.assertEqual(args.sample_fps, 0.1)

    def test_draw_boxes_adds_rectangles_without_text(self) -> None:
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        detections = [{"bboxXyxyPixels": [10, 8, 30, 24]}]

        draw_boxes(frame, detections)

        self.assertTrue(np.any(frame[8, 10:31] != 0))
        self.assertTrue(np.any(frame[24, 10:31] != 0))
        # The interior remains untouched, which distinguishes a box-only
        # export from a filled or text-labelled overlay.
        self.assertTrue(np.all(frame[12:21, 14:27] == 0))


class SamplingFrameIndexesTests(unittest.TestCase):
    def test_exact_ten_second_clip_samples_window_boundary(self) -> None:
        metadata = {"frames": 500, "fps": 50.0}

        self.assertEqual(sampling_frame_indexes(metadata, 0.1), [0])

    def test_partial_window_uses_its_boundary(self) -> None:
        metadata = {"frames": 90, "fps": 30.0}

        self.assertEqual(sampling_frame_indexes(metadata, 0.1), [0])

    def test_long_clip_has_one_sample_per_window(self) -> None:
        metadata = {"frames": 9_000, "fps": 30.0}

        indexes = sampling_frame_indexes(metadata, 0.1)

        self.assertEqual(len(indexes), 30)
        self.assertEqual(indexes[0], 0)
        self.assertEqual(indexes[-1], 8_700)

    def test_full_rate_keeps_every_frame_and_honors_limit(self) -> None:
        metadata = {"frames": 8, "fps": 4.0}

        self.assertEqual(sampling_frame_indexes(metadata, 4.0, 3), [0, 1, 2])

    def test_one_fps_samples_the_start_of_each_second(self) -> None:
        metadata = {"frames": 90, "fps": 30.0}

        self.assertEqual(sampling_frame_indexes(metadata, 1.0), [0, 30, 60])

    def test_detection_samples_stay_sparse_when_export_is_one_fps(self) -> None:
        metadata = {"frames": 900, "fps": 30.0}
        detection_indexes = sampling_frame_indexes(metadata, 0.1)
        render_indexes = sampling_frame_indexes(metadata, 1.0)

        self.assertEqual(len(detection_indexes), 3)
        self.assertEqual(len(render_indexes), 30)
        self.assertEqual(
            render_positions_for_detection_indexes(detection_indexes, render_indexes),
            [0, 10, 20],
        )

    def test_detection_and_export_share_positions_at_ten_second_output(self) -> None:
        metadata = {"frames": 900, "fps": 30.0}
        detection_indexes = sampling_frame_indexes(metadata, 0.1)
        render_indexes = sampling_frame_indexes(metadata, 0.1)

        self.assertEqual(
            render_positions_for_detection_indexes(detection_indexes, render_indexes),
            [0, 1, 2],
        )

    def test_render_timestamps_identify_the_same_source_frames(self) -> None:
        metadata = {"frames": 900, "fps": 30.0}
        render_indexes = sampling_frame_indexes(metadata, 0.1)
        _playback_fps, _holds, timestamps = review_render_plan(
            metadata, 0.1, len(render_indexes)
        )

        self.assertEqual(
            [index / metadata["fps"] for index in render_indexes],
            timestamps,
        )

    def test_sparse_review_matches_sampling_rate_one_to_one(self) -> None:
        metadata = {"frames": 500, "fps": 50.0}

        playback_fps, holds, timestamps = review_render_plan(metadata, 0.1, 1)

        self.assertEqual(playback_fps, 0.1)
        self.assertEqual(holds, [1])
        self.assertEqual(timestamps, [0.0])

    def test_sparse_review_uses_sample_timestamps(self) -> None:
        metadata = {"frames": 750, "fps": 30.0}

        playback_fps, holds, timestamps = review_render_plan(metadata, 0.1, 3)

        self.assertEqual(playback_fps, 0.1)
        self.assertEqual(holds, [1, 1, 1])
        self.assertEqual(timestamps, [0.0, 10.0, 20.0])

    def test_one_fps_review_timestamps_are_one_second_apart(self) -> None:
        metadata = {"frames": 90, "fps": 30.0}

        playback_fps, holds, timestamps = review_render_plan(metadata, 1.0, 3)

        self.assertEqual(playback_fps, 1.0)
        self.assertEqual(holds, [1, 1, 1])
        self.assertEqual(timestamps, [0.0, 1.0, 2.0])

    def test_sparse_review_frame_count_matches_json_records(self) -> None:
        metadata = {"frames": 7_830_000, "fps": 25.0}

        playback_fps, holds, timestamps = review_render_plan(metadata, 0.1, 522)

        self.assertEqual(playback_fps, 0.1)
        self.assertEqual(sum(holds), 522)
        self.assertEqual(len(timestamps), 522)
        self.assertEqual(timestamps[-1], 5_210.0)


class VideoEncoderTests(unittest.TestCase):
    def test_partial_video_lives_outside_export_folder_until_finish(self) -> None:
        class FakeStream:
            def __init__(self, value: bytes = b"") -> None:
                self.value = value

            def close(self) -> None:
                return None

            def read(self) -> bytes:
                return self.value

            def write(self, _value: bytes) -> None:
                return None

        class FakeProcess:
            def __init__(self) -> None:
                self.stdin = FakeStream()
                self.stderr = FakeStream()

            def poll(self) -> int:
                return 0

            def wait(self, **_kwargs: object) -> int:
                return 0

            def terminate(self) -> None:
                return None

            def kill(self) -> None:
                return None

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "exports"
            output_dir.mkdir()
            output = output_dir / "review.mp4"
            process = FakeProcess()
            with patch("backend.inference_runner.subprocess.Popen", return_value=process):
                encoder = VideoEncoder(
                    Path("ffmpeg"),
                    output,
                    width=16,
                    height=16,
                    input_fps=1.0,
                    output_fps=1.0,
                    duration_seconds=0.0,
                    prefer_nvenc=False,
                )
                self.assertNotEqual(encoder.partial.parent.resolve(), output_dir.resolve())
                self.assertEqual(
                    encoder.partial.drive.lower(), output.drive.lower()
                )
                self.assertEqual(list(output_dir.iterdir()), [])
                encoder.partial.write_bytes(b"finished-video")
                result = encoder.finish()

            self.assertEqual(output.read_bytes(), b"finished-video")
            self.assertEqual(result["path"], str(output))
            self.assertEqual(list(output_dir.iterdir()), [output])

    def test_abort_removes_staged_video_without_touching_export_folder(self) -> None:
        class FakeStream:
            def close(self) -> None:
                return None

            def read(self) -> bytes:
                return b""

        class FakeProcess:
            def __init__(self) -> None:
                self.stdin = FakeStream()
                self.stderr = FakeStream()

            def poll(self) -> int:
                return 0

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "exports"
            output_dir.mkdir()
            output = output_dir / "review.mp4"
            with patch(
                "backend.inference_runner.subprocess.Popen",
                return_value=FakeProcess(),
            ):
                encoder = VideoEncoder(
                    Path("ffmpeg"),
                    output,
                    width=16,
                    height=16,
                    input_fps=1.0,
                    output_fps=1.0,
                    duration_seconds=0.0,
                    prefer_nvenc=False,
                )
                staging_directory = encoder.partial.parent
                encoder.partial.write_bytes(b"incomplete-video")
                encoder.abort()

            self.assertFalse(staging_directory.exists())
            self.assertFalse(output.exists())
            self.assertEqual(list(output_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
