import cv2 # type: ignore
import torch # type: ignore
import numpy as np
from torchvision import models, transforms # type: ignore
from collections import deque

# --------------------------------------------------------
# Vehicle Speed Detection using Faster R-CNN and OpenCV
# --------------------------------------------------------

print("🔹 Loading Faster R-CNN model... please wait")
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
print("✅ Model loaded successfully!")

# ----------------------------
# Set up video capture
# ----------------------------
video_path = "your_video.mp4"  # Change to your own video filename
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Could not open video file.")
    exit()

# Get frame rate (FPS)
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"🎥 Video FPS: {fps}")

# Conversion factor: meters per pixel (adjust for accuracy)
distance_per_pixel = 0.05  # approx 5 cm per pixel
track_history = {}
max_history = 10  # number of previous frames to remember

# Transform image for the model
transform = transforms.Compose([
    transforms.ToTensor()
])

# ----------------------------
# Process each video frame
# ----------------------------
print("🚀 Processing video... Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Video processing completed.")
        break

    # Convert to tensor
    image_tensor = transform(frame).unsqueeze(0)

    # Perform object detection
    with torch.no_grad():
        predictions = model(image_tensor)

    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    for i, box in enumerate(boxes):
        if scores[i] > 0.7:  # Confidence threshold
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Assign temporary ID based on detection index
            vehicle_id = i
            if vehicle_id not in track_history:
                track_history[vehicle_id] = deque(maxlen=max_history)
            track_history[vehicle_id].append((cx, cy))

            # Calculate vehicle speed
            if len(track_history[vehicle_id]) >= 2:
                (x_prev, y_prev) = track_history[vehicle_id][-2]
                distance = np.sqrt((cx - x_prev)**2 + (cy - y_prev)**2)
                speed_mps = distance * distance_per_pixel * fps
                speed_kmph = speed_mps * 3.6

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{speed_kmph:.1f} km/h", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Display result
    cv2.imshow("Vehicle Speed Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
