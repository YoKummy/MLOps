Phase 3: Rollback & Recovery (fix-bad-data)

Goal: Demonstrate the "Time Machine" capability of MLOps to rescue a project from corrupted data.

    Key Files: dvc.lock (restored), catvdog.yaml.dvc (restored).

    What this branch does:

        Simulates a "Data Disaster" (deleted or bad data).

        Uses git checkout <hash> -- dvc.lock to grab a "good" state from the past.

        Uses dvc checkout to physically restore deleted images and weights from the local cache.

    Core Command: git checkout <hash> -- dvc.lock followed by dvc checkout.