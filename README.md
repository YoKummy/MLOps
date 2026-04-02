Phase 2: The Training Pipeline (Phase-2-experiment)

Goal: Automate the YOLOv11 training process and track performance metrics.

    Key Files: train.py, dvc.yaml, dvc.lock, runs/.

    What this branch does:

        Defines a reproducible training pipeline in dvc.yaml.

        Trains the YOLOv11 model on the catvdog dataset.

        Generates the first set of results.csv and best.pt weights.

    Core Command: dvc repro.