# Automated MLOps Pipeline: Computer Vision at the Edge

## 🚀 Project Overview

This project demonstrates a complete **Machine Learning Operations (MLOps)** lifecycle, transitioning a raw image dataset into a production-ready API. The system features an automated **"Metric Gatekeeper"** that protects the production environment by only deploying models that statistically outperform the current live version.

The architecture simulates an industrial "Edge" deployment, where models are trained locally and deployed to a specific factory-floor directory (`C:\Shadow_Pipeline_Production`) via a self-hosted CI/CD runner.

## 🛠️ Tech Stack

- **Model:** YOLOv8 (Ultralytics) for high-speed object detection.
- **Data Versioning:** **DVC (Data Version Control)** to manage heavy datasets and model weights without bloating Git.
- **API Serving:** **FastAPI** & **Uvicorn** for high-performance model inference.
- **Automation:** **GitHub Actions** with a **Self-Hosted Runner** for local hardware integration.
- **Environment:** Conda (Python 3.10).

---

## 🏗️ The 5-Phase Architecture

For a deep dive into the engineering decisions and technical hurdles of each phase, please refer to the [**Full Technical Documentation on Notion**](https://www.notion.so/MLOps-3211d0582cb58047a740f70018b892da?source=copy_link).

### 1. Data & Environment Management (Start here! Go to phase 1 branch)

Establishment of a reproducible environment and using DVC as a "data warehouse" to track raw images and training outputs.

### 2. Reproducible Training Pipelines

Integration of `dvc.yaml` to define dependencies. This ensures that the model only retrains if the underlying data or hyperparameters change.

### 3. The "Data Time Machine" (Rollback)

Implementation of a robust rollback strategy. By checking out specific `dvc.lock` states, the system can instantly revert the entire project—including heavy weights—to a known "good" state.

### 4. API Serving (FastAPI)

The model is served via a REST API. Users can upload images to the `/predict` endpoint and receive structured JSON detections (class, confidence, and bounding boxes).

+2

### 5. CI/CD & Automated Gating

The most critical phase: A GitHub Actions pipeline that performs:

1. **Environment Validation:** Ensures the runner has the correct dependencies.
2. **DVC Sync:** Pulls the latest model weights from the local cache.
3. **Sanity Checks:** Verifies the model and API load without errors.
4. **Metric Gating:** A custom script compares the new model's `mAP50` against the production baseline.
5. **Automated Deployment:** Only "superior" models or bug-fixed code are moved to the production folder.

---

## 🚦 Quick Start

1. **Clone the Repo:** `git clone <repo-url>`
2. **Sync Data:** `dvc checkout`
3. **Run API:** ```bash
cd utils
uvicorn fastAPI:app --host 0.0.0.0 --port 8000
4. **Test:** Navigate to `http://127.0.0.1:8000/docs` and use the interactive Swagger UI.

---

## 📈 Metric Comparison Logic

The **Gatekeeper** (located in `utils/gatekeeper.py`) follows strict logic to ensure stability:

- **mAP New > mAP Production:** Full deployment (Model + Code).
- **mAP New == mAP Production:** Partial deployment (Code-only update).
- **mAP New < mAP Production:** Deployment Rejected (Protects the Edge).