from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('./runs/detect/train2/weights/best.pt')  # Path to your trained model

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)

    # Annotate frame with bounding boxes and labels
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cls = int(box.cls)
            label = model.names[cls]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the result
    cv2.imshow('Rock Paper Scissors Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
