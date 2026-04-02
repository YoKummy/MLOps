Phase 4: Model Serving (Phase-4-Serving)

Goal: Transform a static weight file into a live, interactive web service.

    Key Files: utils/fastAPI.py.

    What this branch does:

        Implements a FastAPI server with a /predict endpoint.

        Uses OpenCV to process incoming images and YOLOv8 to generate JSON detections.

        Provides an interactive Swagger UI for testing model inference.

    Core Command: uvicorn fastAPI:app --reload.