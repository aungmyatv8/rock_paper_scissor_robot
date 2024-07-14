import cv2
import time
import random
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('./runs/detect/train2/weights/best.pt')  # Path to your trained model

# Define a function for the countdown
def countdown(t):
    while t:
        print(t)
        time.sleep(1)
        t -= 1
    print("Make your gesture now!")

# Define a function to capture the gesture
def capture_gesture():
    cap = cv2.VideoCapture(0)
    gesture_detected = None
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
                gesture_detected = label

        # Display the result
        cv2.imshow('Rock Paper Scissors Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return gesture_detected

# Define the logic for determining the winner
def determine_winner(user_gesture, ai_gesture):
    if user_gesture == ai_gesture:
        return "It's a tie!"
    elif (user_gesture == 'rock' and ai_gesture == 'scissors') or \
         (user_gesture == 'paper' and ai_gesture == 'rock') or \
         (user_gesture == 'scissors' and ai_gesture == 'paper'):
        return "User wins!"
    else:
        return "AI wins!"

def main():
    print("Get ready to play Rock, Paper, Scissors with AI!")
    countdown(3)

    user_gesture = capture_gesture()

    if user_gesture is None:
        print("Couldn't detect a valid gesture. Try again.")
        return

    ai_gesture = random.choice(['rock', 'paper', 'scissors'])
    print(f"User gesture: {user_gesture}")
    print(f"AI gesture: {ai_gesture}")

    result = determine_winner(user_gesture, ai_gesture)
    print(result)

if __name__ == "__main__":
    main()
