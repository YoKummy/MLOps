Phase 1: Data Foundations (Phase-1-dvc)

Goal: Establish a "Data Warehouse" using DVC to track raw images without bloating Git.

    Key Files: catvdog/, .dvc/, catvdog.yaml.dvc.

    What this branch does: * Initializes DVC for the project.

        Links the raw image dataset to DVC tracking.

        Sets up the .gitignore to ensure heavy data stays out of the cloud.

    Core Command: dvc add catvdog/raw.