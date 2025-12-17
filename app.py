from flask import Flask, request, jsonify, render_template, send_file
import os
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.ndimage import gaussian_filter
import uuid

app = Flask(__name__)

# ======================================================
# PATHS
# ======================================================
BASE_DIR = os.getcwd()
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "Models")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================
# LOAD MODELS (ONCE AT STARTUP)
# ======================================================
plant_model = YOLO(os.path.join(MODEL_DIR, "Plant_Population.pt"))
tassel_model = YOLO(os.path.join(MODEL_DIR, "tassel.pt"))
branch_model = YOLO(os.path.join(MODEL_DIR, "branch.pt"))
tassel_video_model = YOLO(os.path.join(MODEL_DIR, "Tassel_count.pt"))

# ======================================================
# HOME PAGE (HTML UI)
# ======================================================
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# ======================================================
# 1️⃣ PLANT POPULATION (IMAGE)
# ======================================================
@app.route("/plant_population", methods=["POST"])
def plant_population():

    if "image" not in request.files:
        return "Image not provided", 400

    file = request.files["image"]
    img_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.jpg")
    file.save(img_path)

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    results = plant_model(img, conf=0.03)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    density = np.zeros((h, w), dtype=np.float32)
    points = []

    for box in boxes:
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)
        points.append((cx, cy))
        density[cy, cx] = 1

    if points:
        density = gaussian_filter(density, sigma=15)

    count = len(points)

    cv2.putText(
        img, f"Plant Count: {count}",
        (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
        1.5, (255, 255, 255), 3
    )

    out_path = os.path.join(OUTPUT_DIR, f"plant_{uuid.uuid4()}.jpg")
    cv2.imwrite(out_path, img)

    return jsonify({
        "plant_count": count,
        "annotated_image": out_path
    })

# ======================================================
# 2️⃣ TASSEL + BRANCH (IMAGE)
# ======================================================
@app.route("/tassel_branch_image", methods=["POST"])
def tassel_branch_image():

    if "image" not in request.files:
        return "Image not provided", 400

    file = request.files["image"]
    img_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.jpg")
    file.save(img_path)

    img = cv2.imread(img_path)

    tassel_results = tassel_model(img, conf=0.45)
    branch_results = branch_model(img, conf=0.15)

    tassel_count = 0
    branch_count = 0
    branch_id = 1

    # Tassel bounding boxes
    for r in tassel_results:
        for box in r.boxes:
            tassel_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Branch segmentation
    for r in branch_results:
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy()

            for i, mask in enumerate(masks):
                branch_count += 1

                mask = (mask * 255).astype(np.uint8)
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

                overlay = np.zeros_like(img)
                overlay[:, :, 1] = mask

                img = cv2.addWeighted(img, 1.0, overlay, 0.5, 0)

                x1, y1, x2, y2 = boxes[i].astype(int)
                cv2.putText(
                    img, f"id={branch_id}",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2
                )
                branch_id += 1

    summary = f"Tassels: {tassel_count} | Branches: {branch_count}"
    cv2.putText(
        img, summary,
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
        1.2, (0, 255, 255), 3
    )

    out_path = os.path.join(OUTPUT_DIR, f"tassel_branch_{uuid.uuid4()}.jpg")
    cv2.imwrite(out_path, img)

    return jsonify({
        "tassel_count": tassel_count,
        "branch_count": branch_count,
        "annotated_image": out_path
    })

# ======================================================
# 3️⃣ TASSEL VIDEO COUNTING
# ======================================================
@app.route("/tassel_video", methods=["POST"])
def tassel_video():

    if "video" not in request.files:
        return "Video not provided", 400

    file = request.files["video"]
    video_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.mp4")
    file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    top_line_y = int(height * 0.1)
    bottom_line_y = int(height * 0.9)

    prev_centers = {}
    counted_forward = set()
    counted_backward = set()

    forward = 0
    backward = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = tassel_video_model.track(
            frame,
            persist=True,
            conf=0.4,
            iou=0.5,
            verbose=False
        )[0]

        if results.boxes.id is None:
            continue

        ids = results.boxes.id.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cy = (y1 + y2) // 2

            prev_cy = prev_centers.get(track_id)
            prev_centers[track_id] = cy

            if prev_cy is not None:
                if track_id not in counted_forward and prev_cy < top_line_y <= cy:
                    forward += 1
                    counted_forward.add(track_id)

                if track_id not in counted_backward and prev_cy > bottom_line_y >= cy:
                    backward += 1
                    counted_backward.add(track_id)

    cap.release()

    return jsonify({
        "forward": forward,
        "backward": backward,
        "final_unique_count": forward + backward
    })

# ======================================================
# RUN SERVER
# ======================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
