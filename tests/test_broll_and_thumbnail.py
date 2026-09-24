import unittest
from pathlib import Path
from unittest.mock import patch
from src.thumbnail_generator import generate_clip_thumbnail
from src.broll_manager import select_best_video_file, find_broll_cues_for_clip, download_broll_clip, _broll_cache_path
from src.transcriber import WordTimestamp


def fake_loudness_measurement():
    return {
        "input_i": -14.0,
        "input_tp": -1.5,
        "input_lra": 1.0,
        "input_thresh": -24.0,
        "target_offset": 0.0,
    }


def copy_normalized_output(source_path, destination_path, measurement):
    Path(destination_path).write_bytes(Path(source_path).read_bytes())


class TestBrollAndThumbnail(unittest.TestCase):
    def test_thumbnail_generation_fallback_canvas(self):
        """Thumbnail generator should produce a valid JPEG even if video file is dummy."""
        out_path = Path("clips/test_dummy_thumb.jpg")
        try:
            res = generate_clip_thumbnail(
                clip_path=Path("clips/non_existent_clip.mp4"),
                title="TEST VIRAL HOOK TITLE",
                output_path=out_path,
                badge_text="MUST WATCH",
                watermark="@testchannel"
            )
            self.assertIsNotNone(res)
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 1000)
        finally:
            if out_path.exists():
                out_path.unlink()

    def test_thumbnail_uses_duration_middle_frame_and_atomic_output(self):
        import tempfile
        from unittest.mock import patch
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output_path = tmp_path / "nested" / "thumb.jpg"
            frame = Image.new("RGBA", (320, 240), (30, 60, 90, 255))
            with patch("src.thumbnail_generator.extract_frame_at_time", side_effect=[None, frame]) as extract_mock, \
                 patch("src.thumbnail_generator._probe_video_duration", return_value=4.0):
                result = generate_clip_thumbnail(
                    clip_path=tmp_path / "missing.mp4",
                    title="A long title that should stay inside the canvas",
                    output_path=output_path,
                    extract_time=3.0,
                    watermark="",
                )
            self.assertEqual(result, output_path)
            self.assertTrue(output_path.exists())
            self.assertEqual(extract_mock.call_args_list[1].kwargs["timestamp"], 2.0)
            self.assertEqual(list(output_path.parent.glob("*.part")), [])

    def test_broll_select_best_video_file(self):
        """select_best_video_file selects the optimal HD MP4 stream."""
        mock_entry = {
            "video_files": [
                {"file_type": "video/mp4", "width": 540, "height": 960, "link": "https://example.com/sd.mp4"},
                {"file_type": "video/mp4", "width": 1080, "height": 1920, "link": "https://example.com/hd.mp4"},
            ]
        }
        best = select_best_video_file(mock_entry)
        self.assertEqual(best, "https://example.com/hd.mp4")

    def test_broll_cache_path_is_stable(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            first = _broll_cache_path("https://example.com/clip.mp4?x=1", "money", out_dir)
            second = _broll_cache_path("https://example.com/clip.mp4?x=1", "money", out_dir)
            self.assertEqual(first, second)
            self.assertIn("broll_money_", first.name)

    def test_invalid_broll_download_is_not_cached(self):
        import tempfile
        from unittest.mock import MagicMock, patch

        with tempfile.TemporaryDirectory() as tmpdir:
            response = MagicMock(status_code=200)
            response.iter_content.return_value = [b"x" * 20000]
            with patch("src.broll_manager.requests.get", return_value=response), \
                 patch("src.broll_manager.subprocess.run", return_value=MagicMock(returncode=1, stdout="")):
                result = download_broll_clip("https://example.com/clip.mp4", "money", Path(tmpdir))
            self.assertIsNone(result)
            self.assertEqual(list(Path(tmpdir).glob("*.mp4")), [])

    def test_valid_broll_download_is_atomically_cached(self):
        import tempfile
        from unittest.mock import MagicMock, patch

        with tempfile.TemporaryDirectory() as tmpdir:
            response = MagicMock(status_code=200)
            response.iter_content.return_value = [b"x" * 20000]
            with patch("src.broll_manager.requests.get", return_value=response) as get_request, \
                 patch("src.broll_manager.subprocess.run", return_value=MagicMock(returncode=0, stdout="video\n")):
                result = download_broll_clip("https://example.com/clip.mp4", "money", Path(tmpdir))
                self.assertIsNotNone(result)
                self.assertEqual(list(Path(tmpdir).glob("*.part")), [])
                cached = download_broll_clip("https://example.com/clip.mp4", "money", Path(tmpdir))
                self.assertEqual(cached, result)
                get_request.assert_called_once()

    @patch("src.broll_manager.search_pexels_broll")
    @patch("src.broll_manager.download_broll_clip")
    def test_find_broll_cues_for_clip(self, mock_download, mock_search):
        """B-roll cues should match finance keywords within safe time bounds."""
        mock_search.return_value = [{"video_files": [{"file_type": "video/mp4", "height": 1920, "width": 1080, "link": "https://example.com/test.mp4"}]}]
        mock_download.return_value = Path("assets/broll/dummy.mp4")

        words = [
            WordTimestamp(word="hello", start=1.0, end=1.5),
            WordTimestamp(word="we", start=2.0, end=2.5),
            WordTimestamp(word="talk", start=3.0, end=3.5),
            WordTimestamp(word="about", start=4.0, end=4.5),
            WordTimestamp(word="money", start=6.0, end=6.5),
            WordTimestamp(word="today", start=7.0, end=7.5)
        ]
        with patch("src.broll_manager.ENABLE_BROLL", True), \
             patch("src.broll_manager.PEXELS_API_KEY", "test-key"), \
             patch.object(Path, "exists", return_value=True):
            cues = find_broll_cues_for_clip(words, clip_duration=30.0, max_brolls=1)
            self.assertEqual(len(cues), 1)
            self.assertEqual(cues[0][3], "money")
            self.assertEqual(cues[0][0], 6.0)

    def test_video_filtergraph_with_cover_overlay(self):
        """Filtergraph should properly overlay 0.25s cover thumbnail frame."""
        from src.video_editor import build_video_filtergraph
        from src.face_tracker import FramingDecision

        framing = FramingDecision(mode="single_smooth", face_count=1, smoothed_center_x=960)
        fg = build_video_filtergraph(
            framing=framing,
            cover_input_idx=1,
            cover_duration=0.25,
            burn_subtitles=False
        )
        self.assertIn("[1:v]scale=1080:1920", fg)
        self.assertIn("overlay=enable='between(t,0,0.25)'[outv]", fg)

    def test_broll_filtergraph_uses_finite_source_window(self):
        from src.video_editor import build_video_filtergraph
        from src.face_tracker import FramingDecision

        framing = FramingDecision(mode="single_smooth", face_count=1, smoothed_center_x=960)
        fg = build_video_filtergraph(
            framing=framing,
            burn_subtitles=False,
            broll_inputs=[(1, 6.0, 9.5, 6.0)],
        )
        self.assertIn("[1:v]trim=start=6.00:end=9.50,setpts=PTS-STARTPTS", fg)
        self.assertIn(
            "overlay=enable='between(t,6.00,9.50)':eof_action=pass:repeatlast=0",
            fg,
        )

    def test_render_bounds_output_and_builds_broll_input(self):
        import tempfile
        from unittest.mock import MagicMock, patch
        from src.face_tracker import FramingDecision
        from src.video_editor import render_viral_clip

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_path = tmp_path / "source.mp4"
            broll_path = tmp_path / "broll.mp4"
            output_path = tmp_path / "output.mp4"
            source_path.write_bytes(b"source")
            broll_path.write_bytes(b"broll")

            def fake_run(cmd, *args, **kwargs):
                result = MagicMock(returncode=0, stdout="", stderr="")
                if cmd[0] == "ffmpeg":
                    Path(cmd[-1]).write_bytes(b"rendered")
                return result

            with patch("subprocess.run", side_effect=fake_run) as mock_run, \
                 patch("src.video_editor._measure_loudness", return_value=fake_loudness_measurement()), \
                 patch("src.video_editor._apply_measured_loudness", side_effect=copy_normalized_output), \
                 patch("src.video_editor._validate_rendered_output"):
                render_viral_clip(
                    source_video_path=source_path,
                    start_time=0.0,
                    end_time=1.0,
                    output_clip_path=output_path,
                    framing=FramingDecision(
                        mode="single_smooth",
                        face_count=0,
                        video_width=320,
                        video_height=240,
                        active_w=320,
                        active_h=240,
                        smoothed_center_x=160,
                    ),
                    burn_subtitles=False,
                    broll_cues=[(0.25, 0.75, broll_path, "money")],
                )

            ffmpeg_call = next(
                call for call in mock_run.call_args_list
                if call[0][0][0] == "ffmpeg"
            )
            cmd = ffmpeg_call[0][0]
            self.assertIn("-stream_loop", cmd)
            self.assertIn("-t", cmd)
            self.assertEqual(cmd[cmd.index("-t") + 1], "1.00")
            filter_graph = cmd[cmd.index("-filter_complex") + 1]
            self.assertIn("aresample=48000:async=1:first_pts=0", filter_graph)
            self.assertIn("atrim=duration=1.000", filter_graph)
            self.assertIn("apad=whole_dur=1.000", filter_graph)
            self.assertIn("asetpts=PTS-STARTPTS", filter_graph)
            self.assertIn("trim=start=0.25:end=0.75", filter_graph)
            self.assertIn("between(t,0.25,0.75)", filter_graph)

    def test_measured_loudnorm_filter_contains_measurements(self):
        from src.video_editor import _measured_loudnorm_filter

        measured = fake_loudness_measurement()
        filter_text = _measured_loudnorm_filter(measured)
        self.assertIn("measured_I=-14.0", filter_text)
        self.assertIn("measured_TP=-1.5", filter_text)
        self.assertIn("linear=true", filter_text)

    def test_render_rejects_underlength_source_duration(self):
        import tempfile
        from unittest.mock import MagicMock, patch
        from src.face_tracker import FramingDecision
        from src.video_editor import render_viral_clip

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_path = tmp_path / "source.mp4"
            output_path = tmp_path / "output.mp4"
            source_path.write_bytes(b"source")

            def fake_run(cmd, *args, **kwargs):
                if cmd[0] == "ffprobe" and "format=duration" in cmd:
                    return MagicMock(returncode=0, stdout="1.50\n", stderr="")
                return MagicMock(returncode=0, stdout="", stderr="")

            with patch("subprocess.run", side_effect=fake_run):
                with self.assertRaisesRegex(RuntimeError, "does not cover requested clip"):
                    render_viral_clip(
                        source_video_path=source_path,
                        start_time=0.0,
                        end_time=5.0,
                        output_clip_path=output_path,
                        framing=FramingDecision(
                            mode="single_smooth",
                            face_count=0,
                            video_width=320,
                            video_height=240,
                            active_w=320,
                            active_h=240,
                            smoothed_center_x=160,
                        ),
                        burn_subtitles=False,
                    )

    def test_render_adds_silent_voice_bed_for_confirmed_no_audio(self):
        import tempfile
        from unittest.mock import MagicMock, patch
        from src.face_tracker import FramingDecision
        from src.video_editor import render_viral_clip

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_path = tmp_path / "source.mp4"
            output_path = tmp_path / "output.mp4"
            source_path.write_bytes(b"source")

            def fake_run(cmd, *args, **kwargs):
                if cmd[0] == "ffprobe":
                    return MagicMock(returncode=0, stdout="", stderr="")
                Path(cmd[-1]).write_bytes(b"rendered")
                return MagicMock(returncode=0, stdout="", stderr="")

            with patch("subprocess.run", side_effect=fake_run) as mock_run, \
                 patch("src.video_editor._measure_loudness", return_value=fake_loudness_measurement()), \
                 patch("src.video_editor._apply_measured_loudness", side_effect=copy_normalized_output), \
                 patch("src.video_editor._validate_rendered_output"):
                render_viral_clip(
                    source_video_path=source_path,
                    start_time=0.0,
                    end_time=1.0,
                    output_clip_path=output_path,
                    framing=FramingDecision(
                        mode="single_smooth",
                        face_count=0,
                        video_width=320,
                        video_height=240,
                        active_w=320,
                        active_h=240,
                        smoothed_center_x=160,
                    ),
                    burn_subtitles=False,
                )

            ffmpeg_call = next(
                call for call in mock_run.call_args_list
                if call[0][0][0] == "ffmpeg"
            )
            cmd = ffmpeg_call[0][0]
            filter_graph = cmd[cmd.index("-filter_complex") + 1]
            self.assertIn("anullsrc=channel_layout=stereo:sample_rate=48000", cmd)
            self.assertIn("[1:a]", filter_graph)
            self.assertIn("-map", cmd)

    def test_render_fails_when_audio_probe_cannot_classify_source(self):
        import tempfile
        from unittest.mock import MagicMock, patch
        from src.face_tracker import FramingDecision
        from src.video_editor import render_viral_clip

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_path = tmp_path / "source.mp4"
            source_path.write_bytes(b"source")
            probe_failure = MagicMock(returncode=1, stdout="", stderr="probe failed")

            with patch("subprocess.run", return_value=probe_failure) as mock_run:
                with self.assertRaisesRegex(RuntimeError, "Audio probe failed"):
                    render_viral_clip(
                        source_video_path=source_path,
                        start_time=0.0,
                        end_time=1.0,
                        output_clip_path=tmp_path / "output.mp4",
                        framing=FramingDecision(
                            mode="single_smooth",
                            face_count=0,
                            video_width=320,
                            video_height=240,
                            active_w=320,
                            active_h=240,
                            smoothed_center_x=160,
                        ),
                        burn_subtitles=False,
                    )

            self.assertFalse(any(call[0][0][0] == "ffmpeg" for call in mock_run.call_args_list))

if __name__ == "__main__":
    unittest.main()
