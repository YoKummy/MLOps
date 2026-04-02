Phase 5: CI/CD & Production Gating (main)

Goal: The "Final Boss"—Automated testing and safety-gated deployment to the factory floor.

    Key Files: .github/workflows/mlops-ci.yml, utils/gatekeeper.py.

    What this branch does:

        Runs a Self-Hosted Runner on local hardware (PC52525).

        The Gatekeeper: A Python script that compares the new model's mAP against the production high score.

        Continuous Deployment (CD): Automatically updates the C:\Shadow_Pipeline_Production folder only if the model passes the quality gate.

    Core Command: git push (triggers the entire automated pipeline).