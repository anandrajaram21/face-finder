# Contributor notes

## Architecture

The portable pipeline is local: OpenCV YuNet detects faces in gallery and query images, OpenCV SFace generates embeddings, and FAISS indexes/searches embeddings with source-image paths and face coordinates. Model weights are downloaded on first use and cached locally. Keep the CLI (`face-finder index` / `face-finder search`) thin, use the same detection/embedding path for both operations, and keep the pipeline local without credential requirements.

## Development and checks

```sh
uv sync
uv run python -m unittest discover -s tests -v
uv run face-finder index examples/gallery
uv run face-finder search examples/query/millie-query.jpg
```

The CLI lives in `src/face_finder/cli.py`; the standalone Colab demo is `notebooks/demo.ipynb`. Local index data in `.face-finder/` and downloaded weights in `~/.cache/face-finder/models/` must not be committed. Keep the notebook self-contained (it runs directly on Colab, not from a local package). Model downloads must retain SHA-256 verification.

Verify any added example images are decodable, adequately sized for face detection, and licensed for redistribution. The 52-photo gallery includes real group photos; both query photos come from different source images than the gallery. Record **each file's** author, license, and source URL in `examples/README.md`; do not add private photos.
