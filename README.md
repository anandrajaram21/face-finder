# face-finder

Find photos containing a face from a reference image. Designed as a portable, lightweight local pipeline: OpenCV YuNet detects faces, OpenCV SFace makes embeddings, and FAISS searches them. No Pinecone, API credentials, cloud account, or GPU required. Model weights download on first run (internet required then); subsequent runs use the local copies.

## Run locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from this repository:

```sh
uv sync
uv run face-finder index examples/gallery
uv run face-finder search examples/query/barack-obama-2012.jpg
```

The gallery contains Barack and Michelle Obama portraits and a two-person collage; the separate query is a different photograph of Barack Obama. Image licenses and credits are in [examples/README.md](examples/README.md). Replace `examples/gallery` with any folder of photos and the query path with your reference photo. Re-run `index` after changing the gallery; unchanged photos (including no-face photos) are reused, removed photos disappear from results. Results list similarity scores, photo paths and `[x, y, width, height]` face boxes. To search a different gallery independently, pass the same `--index-dir PATH` to both commands. `search --top-k 5 --threshold 0.363` changes the per-face number of results and similarity cutoff (scores are **not** identity probabilities). See `uv run face-finder --help` and `uv run face-finder search --help`.

CPU-only on macOS, Linux and Windows; no GPU required. First run downloads ~39 MB of checked model weights into `~/.cache/face-finder/models`; the Python environment takes additional disk space. OpenCV YuNet + SFace is intentionally lighter than the source notebook's TensorFlow VGG-Face + RetinaFace. Search quality and demographic performance vary; check results yourself before relying on them. Keep the local index and private photos out of Git. Use face matching responsibly and with permission for any other photos you index.

## Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anandrajaram21/face-finder/blob/main/notebooks/demo.ipynb)

The [interactive standalone notebook](notebooks/demo.ipynb) accepts a ZIP of photos or a Google Drive folder and a query image upload; no source edits, API keys or local installation needed. While this GitHub repository is private, the Colab badge requires repository access; anyone with the downloaded notebook can instead upload the `.ipynb` file to Colab. Once the repo is public, the badge works for everyone. Run its cells from top to bottom and click **Build gallery**, then **Search**. The Colab index is in memory and resets with the runtime. [`notebooks/original-colab.ipynb`](notebooks/original-colab.ipynb) is preserved only for provenance; it is incomplete and uses Pinecone. The CLI and notebook use the same YuNet + SFace model family and local FAISS search.
