# Food Detection — 50-Class (MobileNetV2 vs EfficientNetB0)

## **Project**
<img width="1860" height="108" alt="image" src="https://github.com/user-attachments/assets/ad86d0f3-ed8f-46d3-b9e2-5f44ba872872" />

---

## Overview

This project implements an end-to-end food recognition system that:

1. Classifies images into **50 food classes** (examples: pizza, sushi, hamburger, salad, etc.).
2. Looks up nutritional values (calories, protein, fat, carbs per 100 g) from an integrated nutrition database and displays them.

We compare two lightweight, production-friendly CNN feature extractors:

* **MobileNetV2** (mobile-optimized)
* **EfficientNetB0** (compound scaling)

The data source is the [Food-101 dataset (Kaggle)](https://www.kaggle.com/datasets/dansbecker/food-101). The notebook trains both models, visualizes results, and saves models + visual artifacts for deployment.

---

## Highlights

* Trained on **~37,500** images (subset of Food-101 focused on 50 classes).
* Best model achieved **~72% test accuracy** on held-out images (report this exact number in the notebook run output for reproducibility).
* Provides **instant nutrition lookup** (pre-populated `NUTRITION_DB`) after classification.
* Includes a **web app** (folder: `food-calorie-app`) for client-side classification and nutrition display.

---

## Problem Statement

Food recognition is a fine-grained, real-world vision problem with challenges including large intra-class variability (e.g., many styles of pizza) and small inter-class differences (e.g., steak vs pork chop). This project aims to study accuracy vs. efficiency tradeoffs using small, deployable CNNs and to produce a working prototype that maps photos → food label → nutrition facts.

---

## Objectives & Contributions

* Implement two SOTA, efficient backbones and compare them under matched training regimes.
* Provide an end-to-end pipeline: **image preprocessing → model inference → nutrition lookup**.
* Produce visualizations (data distribution, sample predictions, side-by-side model comparison).
* Deliver code, trained models, a web app demo, and a short recorded presentation.

---

## Repository Structure

```
repo-root/
├─ food-calorie-app/           # Web front-end (React / static JS) for demo
│  ├─ model/                   # Converted model and metadata for client inference
│  │  ├─ classes.json
│  │  ├─ group1-shard1of1.bin  # (example names)
│  │  └─ model.json
│  ├─ .gitignore
│  ├─ README.md
│  ├─ index.html
│  ├─ netlify.toml
│  ├─ script.js
│  ├─ style.css
├─ AAI521_team4_FinalProject.ipynb    # Colab notebook: training, eval, visualizations
├─ README.md                   # <- This file
└─ LICENSE
```

> Note: Large binary data (trained weights, full dataset) should be stored in `drive` or on a model storage service (Git LFS, cloud). The notebook shows where outputs are saved (e.g. `SAVE_DIR`).

---

## How to run (quick start)

### Recommended: Run in Google Colab

1. Open `AAI521_team4_FinalProject.ipynb` in Colab. If the notebook mounts Google Drive, grant access so that datasets and outputs are saved there.
2. Make sure the notebook cells that download the Kaggle Food-101 dataset are executed. The notebook includes a helper (e.g. `kagglehub.dataset_download('dansbecker/food-101')`) — ensure Kaggle credentials are configured if required.
3. Execute all cells. The notebook will:

   * prepare datasets for the selected 50 classes
   * build and train MobileNetV2 and EfficientNetB0 models (with pretrained ImageNet weights)
   * save the best checkpoint and final models into `SAVE_DIR`
   * produce visualizations saved to `outputs/`
4. To test inference on a single image, run the visualization/inference cell(s) that call `predict_food(image_path)` and display the top-3 predictions + nutrition lookup.

**Colab tips:**

* Use a GPU runtime. `Runtime > Change runtime type > GPU`.
* If your Colab disconnects, the notebook contains a tiny keep-alive JS snippet (optional).

### Run locally (developer)

**Prerequisites**

* Python 3.8+ (recommend 3.9)
* Virtual environment (venv / conda)
* Enough disk space to download Food-101 (≈ several GB)
* Optional: CUDA GPU and matching TF package for faster training

**Install dependencies**

```bash
git clone <your-repo-url>
cd <repo-root>
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

**Run**

* Open the `Team4_FinalProject.ipynb` with JupyterLab/Notebook or run the provided `.py` training scripts (if available).

### Web app usage

Folder: `food-calorie-app`
The web app demonstrates client-side usage of a converted model (TensorFlow.js / ONNX / other). Files inside:

* `model/`: the converted model files and `classes.json` (mapping indices → class names).
* `index.html`, `app.js`, `style.css`: static site to load a photo, run inference, and show nutrition info.

To run the web demo locally:

```bash
cd food-calorie-app
# Option A: Use a simple static server
python -m http.server 8000
# Then open http://localhost:8000 in your browser
```

---

## Files of interest

* `AAI521_team4_FinalProject.ipynb` — main notebook (data loading, training, visualization, model saving).
* `food-calorie-app/` — static demo app to test models in the browser.
* `outputs/` — saved models, visualizations, JSONs (created after training).
* `nutrition_database.json` — the lookup table used to display nutrition per 100 g.
* `model.json` — final comparison between MobileNetV2 and EfficientNetB0.

---

## Models & Results

* **MobileNetV2** (transfer learning, frozen backbone): saved as `final_model_50class.keras`.
* **EfficientNetB0** (transfer learning, frozen backbone): saved as `final_model_50class_efficientnet.keras`.
* **Best test accuracy** reported during one run: **~72%**. Your accuracy may vary across runs due to randomness in augmentation, initialization, and the exact subset of classes used.

---

## Dependencies (visual)

Below are the primary Python packages used by the notebook. These are shown as badges so you can quickly recognise them.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-brightgreen)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-red)
![Kaggle](https://img.shields.io/badge/Kaggle-API-lightgrey)

**Minimal `requirements.txt` example**

```
tensorflow>=2.6
numpy
matplotlib
pillow
kaggle
scikit-learn
jupyter
```

> Tip: In Colab most packages are preinstalled. Use `pip install -r requirements.txt` when running locally.

---

## Detailed Pipeline & Code Flow

1. **Dataset selection & filtering**

   * Read `meta/train.txt` and `meta/test.txt` and filter to the 50 `SELECTED_CLASSES`.
2. **Preprocessing**

   * Resize images to `224×224` and apply the proper backbone preprocessing (`preprocess_input` for MobileNetV2 and `efficientnet.preprocess_input` for EfficientNetB0).
3. **Dataset pipelines**

   * Build `tf.data.Dataset` pipelines with `shuffle`, `map`, `batch`, and `prefetch` for efficient training.
4. **Model construction**

   * Use pretrained ImageNet backbones (`include_top=False`) → Frozen during initial experiments → Add pooling and a small dense head (512 → 256 → 50 classes).
5. **Training**

   * Train each model with `sparse_categorical_crossentropy` and `Adam` optimizer.
   * Save best checkpoint via `ModelCheckpoint(monitor='val_accuracy')`.
6. **Evaluation & Visualization**

   * Evaluate on held-out test set and produce multiple plots: class distribution, sample predictions, top-3 detailed predictions, model comparison charts.
7. **Export**

   * Save final Keras models and `nutrition_database.json` for downstream use.

---

## Evaluation & Metrics

The notebook reports:

* **Test accuracy** (primary metric)
* **Test loss**
* Visual confusion (via top-3 examples and side-by-side predictions)
* Model parameter counts and approximate training times

When reporting numbers in the README or presentation, always include the exact command and random seed (if set) to allow reproduction.

---

## Troubleshooting & Tips

* If training is slow or runs out of memory: use smaller batch size or enable mixed precision (for GPUs that support it).
* If accuracy is low for some classes: consider class rebalancing or targeted data augmentation for underrepresented classes (rotations, color jitter, random crop).
* If dataset download fails: ensure Kaggle API token is placed at `~/.kaggle/kaggle.json` or use direct dataset upload to Drive.
* If the web demo cannot load the model: ensure the model conversion to TF.js/ONNX was performed and files are placed under `food-calorie-app/model/`.

---

## Recommendations for future work

* Fine-tune backbone layers (unfreeze and retrain) with lower learning rate — often improves accuracy significantly.
* Use more aggressive augmentation and/or class-balanced sampling.
* Explore lightweight ensembling (average predictions from MobileNetV2 and EfficientNetB0) for better robustness.
* Replace the static nutrition lookup with an API-backed nutrition database for more detailed serving sizes and micronutrients.

---

## Presentation & Deliverables

* **Technical report**: Write the full report using the STAR format (Situation, Task, Action, Results). Attach the report and appendices to the repo (or `drive`).
* **GitHub repo**: Push code (without large binaries) to GitHub. Provide instructions to reproduce using Colab.
* **Video**: Record a short demo + explanation including the notebook walkthrough and web demo.

---

## Licensing & Credits

* **Dataset:** [Food-101 on Kaggle](https://www.kaggle.com/datasets/kmader/food41) — originally by Bossard et al.  
  Please refer to the dataset page for license terms and usage guidelines.

* **Code & Models:** Developed by [Team4/ Applied AI CV Course @USD].  
  This repository is released under the **MIT License** — see the included `LICENSE` file for details.  
  MIT License allows you to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, with proper attribution.


---

*Happy experimenting — snap a picture, see the nutrition!*
