# Dog Breed Identification

A web app that identifies a dog's breed from a photo. A FastAPI backend serves an EfficientNetV2B0 model fine-tuned on 120 breeds, with a simple HTML/JS frontend for uploading images and viewing predictions.

## Overview

Upload a photo of a dog, and the model returns its best guess at the breed along with a confidence score. The model was trained on the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) (120 breed classes), using transfer learning on top of EfficientNetV2B0.

## Features

- Image upload through a browser interface
- Breed prediction with a confidence score
- REST API (`/predict`) that can be called independently of the frontend
- Covers 120 dog breeds

## Tech stack

| Layer | Tool |
|---|---|
| Model | EfficientNetV2B0 (via TensorFlow Hub), fine-tuned with `tf_keras` |
| Backend | FastAPI + Uvicorn |
| Image processing | TensorFlow (`tf.image`), Pillow, NumPy |
| Frontend | HTML, CSS, vanilla JS |

## Project structure

```
Dog-Breed-Identification/
├── main.py              # FastAPI app: serves the frontend and the /predict endpoint
├── index.html            # Upload interface served at "/"
├── dog_breed.ipynb        # Notebook: data prep, training, and evaluation of the model
├── fastapi.ipynb          # Notebook: prototyping the FastAPI serving logic
├── requirements.txt        # Python dependencies
└── models/               # Trained model file(s) (.h5) — not tracked in git, see below
```

## Model details

- **Architecture:** EfficientNetV2B0 backbone (loaded through `tensorflow_hub`), fine-tuned on top with a classification head for 120 classes.
- **Input:** RGB images, resized to 224×224, pixel values scaled to the 0–1 range.
- **Output:** A softmax distribution over 120 breed classes; the app returns the highest-probability class and its score.
- **Classes:** the full list of 120 breed names lives in `main.py` (`class_names`) and mirrors the Stanford Dogs Dataset labels (e.g. `golden_retriever`, `siberian_husky`, `yorkshire_terrier`).

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/OmarMarwan1/Dog-Breed-Identification.git
cd Dog-Breed-Identification
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add the trained model

`main.py` loads the model from a path like:

```
models/<your-model-file>.h5
```

Place your trained `.h5` file (produced by `dog_breed.ipynb`) in a `models/` folder in the project root, and make sure the path in `main.py` (`tf_keras.models.load_model(...)`) points to it. This file isn't included in the repo since trained weights are typically too large for git — consider Git LFS or a release asset if you want to version it.

### 4. Run the app

```bash
uvicorn main:app --reload
```

Then open **http://127.0.0.1:8000** in your browser to use the upload interface.

## API reference

### `POST /predict`

Accepts a multipart form upload and returns a breed prediction.

**Request:** `multipart/form-data` with a `file` field containing an image (JPEG).

**Response:**
```json
{
  "filename": "dog.jpg",
  "breed": "golden_retriever",
  "confidence": 92.14
}
```

- `breed` — the predicted class name, matching the raw labels in `class_names` (lowercase, underscores).
- `confidence` — the model's confidence in that prediction, as a percentage (0–100).

> **Note:** if you swap in a different frontend, double check its expected field names against the response above — some earlier versions of the interface expected `predicted_breed` and a `confidence` fraction (0–1) instead. Keep both sides in sync.

## Notebooks

- **`dog_breed.ipynb`** — data loading, preprocessing, model training, and evaluation for the EfficientNetV2B0 classifier.
- **`fastapi.ipynb`** — exploratory notebook used to work out the FastAPI serving logic before it was moved into `main.py`.

## Roadmap ideas

- [ ] Add Git LFS (or a download step) for the model weights so the repo is fully reproducible
- [ ] Add top-3 predictions instead of just the top-1
- [ ] Add automated tests for the `/predict` endpoint
- [ ] Containerize with Docker for easier deployment

## NOTE: the index.html was made by Claude because I don't have experience in developing web pages

## License

No license specified yet — add one (e.g. MIT) if you want others to reuse this freely.
