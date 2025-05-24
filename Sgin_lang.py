from ultralytics import YOLO
import cv2

# Load your custom model
model = YOLO('yolov8n_sgin_detector.pt')

# Open webcam (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Check if webcam is opened
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop through frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    flipped_frame = cv2.flip(frame, 1)  
    # Run inference on the frame
    results = model(flipped_frame,conf = 0.7,agnostic_nms=True)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow('YOLOv8 Webcam Detection', annotated_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
