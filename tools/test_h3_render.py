"""Guard against downloading a reference-video preview as the generated shot."""
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from h3_render import fetch_video


class FetchVideoTests(unittest.TestCase):
    def test_downloads_generated_video_after_input_preview(self):
        entry = {"outputs": {
            "load": {"images": [{"filename": "reference.mp4", "subfolder": "", "type": "input"}]},
            "save": {"images": [{"filename": "generated.mp4", "subfolder": "film", "type": "output"}]},
        }}

        def download(url, **kwargs):
            query = parse_qs(urlparse(url).query)
            payload = b"generated shot" if query["type"] == ["output"] else b"reference preview"
            return io.BytesIO(payload)

        with tempfile.TemporaryDirectory() as directory, patch("h3_render.urllib.request.urlopen", download):
            target = Path(directory) / "shot.mp4"
            fetch_video(entry, target)
            self.assertEqual(target.read_bytes(), b"generated shot")

    def test_input_preview_cannot_satisfy_missing_render(self):
        entry = {"outputs": {
            "load": {"videos": [{"filename": "reference.mp4", "subfolder": "", "type": "input"}]},
        }}
        with tempfile.TemporaryDirectory() as directory, patch(
            "h3_render.urllib.request.urlopen", return_value=io.BytesIO(b"reference preview")
        ):
            target = Path(directory) / "shot.mp4"
            with self.assertRaises(RuntimeError):
                fetch_video(entry, target)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
