import os
import cv2
import time
import uuid
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Fix OpenMP error
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Create folder to save missing objects
if not os.path.exists('missing_objects'):
    os.makedirs('missing_objects')

# Load YOLOv8n and DeepSORT tracker
model = YOLO('yolov8n.pt')
tracker = DeepSort(max_age=10)

# Open webcam
cap = cv2.VideoCapture(0)

prev_objects = {}  # {id: (bbox, class_id)}
missing_counter = 0  # Total missing objects
per_class_missing_counter = {}  # Missing count per class

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Get original frame size
    height, width = frame.shape[:2]

    # Resize for YOLO
    resized_frame = cv2.resize(frame, (640, 640))

    # Object detection
    results = model(resized_frame, conf=0.3, device='cuda:0' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu')
    detections = []
    for r in results:
        boxes = r.boxes.xywh.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        for box, score, cls in zip(boxes, scores, classes):
            # Save each detection's box (no rescaling yet)
            x_center, y_center, w, h = box
            x1 = int((x_center - w / 2) * width / 640)
            y1 = int((y_center - h / 2) * height / 640)
            x2 = int((x_center + w / 2) * width / 640)
            y2 = int((y_center + h / 2) * height / 640)
            detections.append(([x1, y1, x2, y2], score, cls))

    # Object tracking
    tracks = tracker.update_tracks(detections, frame=resized_frame)
    current_objects = {}  # {id: (bbox, class_id)}

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_tlbr().astype(int)  # [x1, y1, x2, y2]
        class_id = int(track.det_class) if hasattr(track, 'det_class') else 0

        # Clip the bounding box to ensure it stays within the image
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)

        current_objects[track_id] = (bbox, class_id)

        # Draw bounding box (without class names)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
        cv2.putText(frame, f'ID:{track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Detect missing objects
    missing_ids = set(prev_objects.keys()) - set(current_objects.keys())

    for mid in missing_ids:
        bbox, class_id = prev_objects[mid]
        x1, y1, x2, y2 = bbox

        crop = frame[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
        if crop.size != 0:
            filename = f'missing_objects/{uuid.uuid4().hex}.jpg'
            cv2.imwrite(filename, crop)

            missing_counter += 1

            if 'unknown' not in per_class_missing_counter:
                per_class_missing_counter['unknown'] = 0
            per_class_missing_counter['unknown'] += 1

    # Update previous frame objects
    prev_objects = current_objects

    # Display counters
    cv2.putText(frame, f'Missing Total: {missing_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display missing counts for top 3 (just using 'unknown' class here)
    y_offset = 70
    for idx, (cls_name, count) in enumerate(sorted(per_class_missing_counter.items(), key=lambda x: x[1], reverse=True)[:3]):
        cv2.putText(frame, f'{cls_name}: {count}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        y_offset += 30

    # Display FPS
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time if elapsed_time > 0 else 0
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show the final frame
    cv2.imshow('Real-Time Full Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
