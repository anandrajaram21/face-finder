# face-finder

Find photos containing a face from a reference image. OpenCV YuNet detects faces, SFace embeds them, and a local FAISS index searches the embeddings. No GPU, API key, Pinecone, or cloud account is needed. Scores are cosine similarities, **not** identity probabilities.

## Local setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and clone this repository. uv installs the required Python version (3.11–3.13) and dependencies from `uv.lock`:

   ```sh
   git clone git@github.com:anandrajaram21/face-finder.git
   cd face-finder
   uv sync --locked
   ```

   If you use HTTPS instead of SSH, clone `https://github.com/anandrajaram21/face-finder.git`. This repository may require GitHub access while it is private.

2. Index a gallery, then search using a **different** photo of a person in that gallery:

   ```sh
   uv run face-finder index examples/gallery
   uv run face-finder search examples/query/millie-query.jpg
   uv run face-finder search examples/query/zendaya-query.jpg
   ```

The 52-photo gallery includes individual and group photos; each query comes from a separate source image. See [image credits and licenses](examples/README.md) for all 54 files. Substitute your own gallery folder and query image. Supported formats: JPEG, PNG, WebP, BMP. The index lives in `.face-finder/` in the current directory; rerun `index` after changing the gallery. Unchanged photos, including those without detectable faces, are reused; removed photos disappear from results. Results show photo paths, similarity scores, and `[x, y, width, height]` face boxes. Use the same `--index-dir PATH` on both commands to keep galleries separate. `search --top-k 5 --threshold 0.363` controls the number of photos per query face and minimum score. Run `uv run face-finder --help` or `uv run face-finder search --help` for options.

The first run needs internet to download ~39 MB of SHA-256-verified model weights from OpenCV Zoo into `~/.cache/face-finder/models/`; later runs work offline. The Python environment needs additional space. Use `--models-dir PATH` **before** `index` or `search` to change the model cache. If an index is incomplete or incompatible, delete its `.face-finder/` directory (or your custom index directory) and re-index. For a failed model download, check connectivity and retry; corrupted cached weights are checked and re-downloaded. Neither cache nor local index is committed.

Run the offline checks with `uv run python -m unittest discover -s tests -v`. The CLI runs on macOS, Linux, and Windows without a GPU. YuNet + SFace is lighter than the historical TensorFlow VGG-Face + RetinaFace prototype; quality and demographic performance vary. Review matches yourself, and only process photos you have permission to use.

## Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anandrajaram21/face-finder/blob/main/notebooks/demo.ipynb)

The [standalone interactive notebook](notebooks/demo.ipynb) runs directly in Colab: run cells top to bottom, upload a ZIP of gallery photos (or mount a Drive folder), click **Build gallery**, upload a query photo, then click **Search**. To use the included examples, ZIP `examples/gallery/` and upload it, then upload either photo in `examples/query/`. Files uploaded via Colab's Files sidebar can instead be entered by their `/content/...` paths. It installs its own dependencies; no local package or API keys are needed. Its index is in memory and resets with the Colab runtime. If the GitHub repository is private, the badge requires access; alternatively download `notebooks/demo.ipynb` and upload it to Colab. [`notebooks/original-colab.ipynb`](notebooks/original-colab.ipynb) is an incomplete historical prototype using Pinecone, not the portable implementation.
