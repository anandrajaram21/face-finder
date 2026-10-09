"""Check the Colab ZIP input without importing Colab-only dependencies."""

import ast
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile


class GalleryArchiveTest(unittest.TestCase):
    def test_zip_from_widget_or_colab_files(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] / "notebooks/demo.ipynb").read_text())
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
        tree = ast.parse(source)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"uploaded_bytes", "gallery_archive"}]
        namespace = {"BytesIO": BytesIO, "Path": Path, "zipfile": zipfile}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "demo.ipynb", "exec"), namespace)

        data = BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr("photo.jpg", b"image")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "gallery.zip"
            path.write_bytes(data.getvalue())
            for upload, fallback in [({"value": ()}, str(path)), ({"value": ({"name": "gallery.zip", "content": data.getvalue()},)}, "")]:
                with namespace["gallery_archive"](type("Upload", (), upload)(), fallback) as archive:
                    self.assertEqual(archive.read("photo.jpg"), b"image")
            with self.assertRaisesRegex(ValueError, "Upload a gallery ZIP"):
                namespace["gallery_archive"](type("Upload", (), {"value": ()})(), "")


if __name__ == "__main__":
    unittest.main()
