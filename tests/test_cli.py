"""Small offline check for incremental FAISS indexing and multi-face search."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from face_finder.cli import build, load, search


class StubFaces:
    def __init__(self):
        self.calls = []

    def extract(self, path):
        self.calls.append(path.name)
        vector = np.zeros(128, dtype="float32")
        vector[0 if "person" in path.name else 1] = 1
        return [(vector, [0, 0, 10, 10])] if "empty" not in path.name else []


class IndexTest(unittest.TestCase):
    def test_incremental_and_search(self):
        with TemporaryDirectory() as temp:
            gallery, cache = Path(temp) / "gallery", Path(temp) / "cache"
            gallery.mkdir()
            for name in ("person.jpg", "other.jpg", "empty.jpg"):
                (gallery / name).write_bytes(b"image")
            faces = StubFaces()
            with redirect_stdout(StringIO()):
                build(gallery, cache, faces)
                build(gallery, cache, faces)
            self.assertEqual(len(faces.calls), 3)  # includes a no-face photo
            self.assertEqual(load(cache)[0].ntotal, 2)
            output = StringIO()
            with redirect_stdout(output):
                search(Path("query-person.jpg"), cache, faces, 5, .363)
            self.assertIn("person.jpg", output.getvalue())
            self.assertNotIn("other.jpg", output.getvalue())
            (gallery / "person.jpg").unlink()
            with redirect_stdout(StringIO()):
                build(gallery, cache, faces)
            self.assertEqual(load(cache)[0].ntotal, 1)


if __name__ == "__main__":
    unittest.main()
