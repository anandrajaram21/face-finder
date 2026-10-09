"""Index a photo folder, then search it with a reference photo."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from urllib.request import urlopen

import cv2
import faiss
import numpy as np

MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "face_detection_yunet", "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
    ),
    "face_recognition_sface_2021dec.onnx": (
        "face_recognition_sface", "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
    ),
}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# Keep the source revision fixed; SHA-256 below also verifies the downloaded bytes.
MODEL_BASE = "https://media.githubusercontent.com/media/opencv/opencv_zoo/47534e27c9851bb1128ccc0102f1145e27f23f98/models"


def model_file(name, cache):
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / name
    folder, digest = MODELS[name]
    if path.exists():
        with path.open("rb") as existing:
            if hashlib.file_digest(existing, "sha256").hexdigest() == digest:
                return path
    url = f"{MODEL_BASE}/{folder}/{name}"
    print(f"Downloading {name} (once)...", file=sys.stderr)
    with tempfile.NamedTemporaryFile(dir=cache, delete=False) as temp:
        temporary = Path(temp.name)
        try:
            with urlopen(url, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    temp.write(chunk)
            temp.flush()
            with temporary.open("rb") as downloaded:
                if hashlib.file_digest(downloaded, "sha256").hexdigest() != digest:
                    raise ValueError(f"Model download failed checksum: {name}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


class Faces:
    def __init__(self, cache):
        detector = model_file("face_detection_yunet_2023mar.onnx", cache)
        recognizer = model_file("face_recognition_sface_2021dec.onnx", cache)
        self.detector = cv2.FaceDetectorYN.create(str(detector), "", (320, 320))
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer), "")

    def extract(self, path):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("Cannot read image")
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, detected = self.detector.detect(image)
        results = []
        for face in [] if detected is None else detected:
            aligned = self.recognizer.alignCrop(image, face)
            vector = self.recognizer.feature(aligned).reshape(-1).astype("float32")
            faiss.normalize_L2(vector.reshape(1, -1))
            results.append((vector, [int(v) for v in face[:4]]))
        return results


def files(folder):
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)


def index_paths(destination):
    return destination / "faces.faiss", destination / "faces.json"


def load(destination):
    vectors_path, manifest_path = index_paths(destination)
    if not vectors_path.exists() or not manifest_path.exists():
        return None, None
    index = faiss.read_index(str(vectors_path))
    manifest = json.loads(manifest_path.read_text())
    if index.ntotal != len(manifest["faces"]) or index.d != 128 or manifest.get("model") != "sface-2021dec-yunet-2023mar":
        raise ValueError("Index and metadata do not match; delete the index directory and rebuild")
    return index, manifest


def build(folder, destination, faces):
    folder = folder.resolve()
    if not folder.is_dir():
        raise ValueError(f"Not a folder: {folder}")
    destination.mkdir(parents=True, exist_ok=True)
    previous, old = load(destination)
    old_by_path = {}
    old_stamps = {}
    if old and old.get("folder") == str(folder):
        old_stamps = old.get("photos", {})
        for i, face in enumerate(old["faces"]):
            old_by_path.setdefault(face["path"], []).append((i, face))
    index = faiss.IndexFlatIP(128)
    records = []
    stamps = {}
    count = 0
    for path in files(folder):
        # Never index the query or cached models if the output directory is inside the gallery.
        if destination.resolve() in path.resolve().parents:
            continue
        relative = str(path.relative_to(folder))
        stamp = [path.stat().st_size, path.stat().st_mtime_ns]
        reused = old_by_path.get(relative, [])
        if old_stamps.get(relative) == stamp and previous:
            found = [(previous.reconstruct(i), record["box"]) for i, record in reused]
        else:
            try:
                found = faces.extract(path)
            except (ValueError, cv2.error) as exc:
                print(f"Skipping {path}: {exc}", file=sys.stderr)
                continue
            count += 1
        stamps[relative] = stamp
        for vector, box in found:
            index.add(np.asarray(vector, dtype="float32").reshape(1, -1))
            records.append({"path": relative, "box": box})
    vectors_path, manifest_path = index_paths(destination)
    # Write index first and manifest last: an interrupted write fails the consistency check.
    with tempfile.NamedTemporaryFile(dir=destination, delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        faiss.write_index(index, str(temporary))
        temporary.replace(vectors_path)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {"folder": str(folder), "model": "sface-2021dec-yunet-2023mar", "photos": stamps, "faces": records}
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination, delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        temporary.write_text(json.dumps(manifest), encoding="utf-8")
        temporary.replace(manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Indexed {len(records)} faces across {len({r['path'] for r in records})} photos ({count} processed; unchanged photos reused).")


def search(image, destination, faces, top_k, threshold):
    index, manifest = load(destination)
    if index is None:
        raise ValueError(f"No index at {destination}; run 'face-finder index' first")
    if index.ntotal == 0:
        print("The gallery contains no detected faces.")
        return
    queries = faces.extract(image)
    if not queries:
        print("No faces detected in query photo.")
        return
    for number, (vector, box) in enumerate(queries, 1):
        print(f"Query face {number} at {box}:")
        scores, ids = index.search(vector.reshape(1, -1), index.ntotal)
        seen = set()
        for score, idx in zip(scores[0], ids[0]):
            record = manifest["faces"][int(idx)]
            if float(score) < threshold:
                break
            if record["path"] in seen:
                continue
            seen.add(record["path"])
            print(f"  {score:.3f}  {Path(manifest['folder']) / record['path']}  face {record['box']}")
            if len(seen) >= top_k:
                break
        if not seen:
            print("  No matches above threshold.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path.home() / ".cache" / "face-finder" / "models", help="Model download cache")
    sub = parser.add_subparsers(dest="action", required=True)
    indexing = sub.add_parser("index", help="Index or update all faces in a folder")
    indexing.add_argument("gallery", type=Path)
    indexing.add_argument("--index-dir", type=Path, default=Path(".face-finder"))
    querying = sub.add_parser("search", help="Find photos matching faces in a query photo")
    querying.add_argument("query", type=Path)
    querying.add_argument("--index-dir", type=Path, default=Path(".face-finder"))
    querying.add_argument("--top-k", type=int, default=5)
    querying.add_argument("--threshold", type=float, default=0.363, help="Minimum cosine similarity, default 0.363 (SFace reference threshold)")
    args = parser.parse_args(argv)
    if args.action == "search" and (args.top_k < 1 or not -1 <= args.threshold <= 1):
        parser.error("--top-k must be positive and --threshold must be between -1 and 1")
    try:
        engine = Faces(args.models_dir)
        if args.action == "index":
            build(args.gallery, args.index_dir, engine)
        else:
            search(args.query, args.index_dir, engine, args.top_k, args.threshold)
    except (ValueError, OSError, RuntimeError, cv2.error) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
