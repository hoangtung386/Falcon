from __future__ import annotations

import os
import tempfile

from falcon.data.io import InputType, get_input_type


class TestInputType:
    def test_image_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            assert get_input_type(f.name) == InputType.IMAGE

    def test_video_file(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            assert get_input_type(f.name) == InputType.VIDEO

    def test_image_folder(self, tmp_path):
        d = tmp_path / "images"
        d.mkdir()
        assert get_input_type(str(d)) == InputType.IMAGE

    def test_video_stream(self):
        assert get_input_type("http://example.com/stream") == InputType.VIDEO_STREAM

    def test_image_url(self):
        assert get_input_type("http://example.com/photo.jpg") == InputType.IMAGE

    def test_unknown_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nonexistent.txt")
            try:
                get_input_type(path)
                assert False, "Should raise ValueError"
            except ValueError:
                pass
