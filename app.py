from pathlib import Path
import time
import cv2
import torch

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    "models/best.pt"
)

UPLOAD_FOLDER = Path("static/uploads")
RESULT_FOLDER = Path("static/results")

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp",
    "bmp",
}

CONFIDENCE = 0.25
IMAGE_SIZE = 640
DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ============================================================
# APP SETUP
# ============================================================

app = Flask(__name__)

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)


# ============================================================
# LOAD MODEL ONCE
# ============================================================

print("\nLoading YOLOv8s model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")
print(f"Model: {MODEL_PATH}")


# ============================================================
# HELPERS
# ============================================================

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():

    # --------------------------------------------------------
    # SHOW PAGE
    # --------------------------------------------------------

    if request.method == "GET":
        return render_template("index.html")


    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if "image" not in request.files:
        return render_template(
            "index.html",
            error="Please select an image."
        )

    file = request.files["image"]

    if file.filename == "":
        return render_template(
            "index.html",
            error="Please select an image."
        )

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            error="Unsupported image format."
        )


    # --------------------------------------------------------
    # SAVE UPLOADED IMAGE
    # --------------------------------------------------------

    filename = secure_filename(file.filename)

    input_path = UPLOAD_FOLDER / filename

    file.save(input_path)


    # --------------------------------------------------------
    # RUN YOLO
    # --------------------------------------------------------

    start_time = time.perf_counter()

    results = model.predict(
        source=str(input_path),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        device=DEVICE,
        save=False,
        verbose=False,
    )

    end_time = time.perf_counter()

    processing_time = (
        end_time - start_time
    ) * 1000


    # --------------------------------------------------------
    # PROCESS RESULTS
    # --------------------------------------------------------

    detections = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])

            confidence = float(
                box.conf[0]
            )

            class_name = model.names[class_id]

            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(
                    confidence * 100,
                    2
                ),
            })


    # --------------------------------------------------------
    # CREATE AND SAVE ANNOTATED IMAGE
    # --------------------------------------------------------

    output_dir = (
        RESULT_FOLDER / "prediction"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_filename = (
        Path(filename).stem + "_result.jpg"
    )

    output_path = (
        output_dir / output_filename
    )


    # Generate annotated image
    annotated_image = results[0].plot()


    # Save annotated image
    cv2.imwrite(
        str(output_path),
        annotated_image
    )


    # Path used by Flask/static
    result_image = (
        f"results/prediction/"
        f"{output_filename}"
    )


    # --------------------------------------------------------
    # RETURN RESULTS TO WEB PAGE
    # --------------------------------------------------------

    return render_template(
        "index.html",
        result_image=result_image,
        detections=detections,
        inference_time=round(
            processing_time,
            2
        ),
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )