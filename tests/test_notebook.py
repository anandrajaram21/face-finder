"""Check the Colab ZIP input without importing Colab-only dependencies."""

import ast
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile


class NotebookUploadTest(unittest.TestCase):
    def test_zip_and_query_from_widget_or_colab_files(self):
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

            query = Path(directory) / "query.jpg"
            query.write_bytes(b"photo")
            empty = type("Upload", (), {"value": ()})()
            self.assertEqual(namespace["uploaded_bytes"](empty, f" {query} "), ("query.jpg", b"photo"))
            widget = type("Upload", (), {"value": ({"name": "widget.jpg", "content": b"widget"},)})()
            self.assertEqual(namespace["uploaded_bytes"](widget, str(query)), ("widget.jpg", b"widget"))
            self.assertEqual(namespace["uploaded_bytes"](empty), (None, None))
            with self.assertRaises(FileNotFoundError):
                namespace["uploaded_bytes"](empty, str(query.with_name("missing.jpg")))


if __name__ == "__main__":
    unittest.main()
