import tkinter as tk
from tkinter import ttk
import cv2
import time
import random
from PIL import Image, ImageTk
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('./runs/detect/train2/weights/best.pt')  # Path to your trained model

# Initialize the camera
cap = cv2.VideoCapture(0)

# Define a function for the countdown
def countdown(t, label):
    for i in range(t, 0, -1):
        label.config(text=str(i))
        label.update()
        time.sleep(1)
    label.config(text="Make your gesture now!")
    label.update()
    time.sleep(1)

# Function to update the video feed
def update_video():
    ret, frame = cap.read()
    if ret:
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

        # Convert the frame to a format suitable for tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the UI with the video frame
        lbl_video.imgtk = imgtk
        lbl_video.config(image=imgtk)
    
    # Schedule the next update
    lbl_video.after(10, update_video)

# Define a function to capture the gesture
def capture_gesture():
    gesture_detected = None
    for _ in range(50):  # Loop to give some time for gesture detection
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
                print(f"Detected gesture: {gesture_detected}")  # Debug print

        if gesture_detected:
            break

    return gesture_detected

# Define the logic for determining the winner
def determine_winner(user_gesture, ai_gesture):
    print("gestures", user_gesture, ai_gesture) # Debug print
    if user_gesture.lower() == ai_gesture:
        return "It's a tie!"
    elif (user_gesture.lower() == 'rock' and ai_gesture == 'scissors') or \
         (user_gesture.lower() == 'paper' and ai_gesture == 'rock') or \
         (user_gesture.lower() == 'scissors' and ai_gesture == 'paper'):
        return "User wins!"
    else:
        return "AI wins!"

# Start the game
def start_game():
    lbl_result.config(text="")
    lbl_ai_gesture.config(text="")
    countdown(3, lbl_countdown)
    
    user_gesture = capture_gesture()
    if user_gesture is None:
        lbl_result.config(text="Couldn't detect a valid gesture. Try again.")
        return
    
    ai_gesture = random.choice(['rock', 'paper', 'scissors'])
    print(f"User gesture: {user_gesture}")  # Debug print
    print(f"AI gesture: {ai_gesture}")      # Debug print
    lbl_ai_gesture.config(text=f"AI gesture: {ai_gesture}")

    result = determine_winner(user_gesture, ai_gesture)
    lbl_result.config(text=f"User gesture: {user_gesture}\n{result}")

# Create the main window
root = tk.Tk()
root.title("Rock Paper Scissors Game")

# Create and place the video label
lbl_video = ttk.Label(root)
lbl_video.grid(row=0, column=0, columnspan=2)

# Create and place the countdown label
lbl_countdown = ttk.Label(root, text="", font=("Helvetica", 24))
lbl_countdown.grid(row=1, column=0, columnspan=2)

# Create and place the AI gesture label
lbl_ai_gesture = ttk.Label(root, text="", font=("Helvetica", 18))
lbl_ai_gesture.grid(row=2, column=0, columnspan=2)

# Create and place the result label
lbl_result = ttk.Label(root, text="", font=("Helvetica", 18))
lbl_result.grid(row=3, column=0, columnspan=2)

# Create and place the start button
btn_start = ttk.Button(root, text="Start Game", command=start_game)
btn_start.grid(row=4, column=0, columnspan=2)

# Start the video feed
update_video()

root.mainloop()

# Release the camera when the window is closed
cap.release()
