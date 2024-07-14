from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8n.pt')  # You can use a different YOLOv8 model if needed

# Define augmentation parameters
augmentation_params = {
    'hsv_h': 0.015,  # Hue augmentation (fraction)
    'hsv_s': 0.7,    # Saturation augmentation (fraction)
    'hsv_v': 0.4,    # Value augmentation (fraction)
    'degrees': 0.2,  # Image rotation (+/- degrees)
    'translate': 0.1,  # Image translation (+/- fraction)
    'scale': 0.5,     # Image scale (+/- gain)
    'shear': 0.0,     # Image shear (+/- degrees)
    'perspective': 0.0,  # Image perspective (+/- fraction), range 0-0.001
    'flipud': 0.0,    # Image flip upside-down (probability)
    'fliplr': 0.5,    # Image flip left-right (probability)
    'mosaic': 1.0,    # Mosaic augmentation (probability)
    'mixup': 0.0,     # MixUp augmentation (probability)
    'copy_paste': 0.0  # Copy-paste augmentation (probability)
}

# Train the model with augmentations
model.train(
    data='./dataset/data.yaml', 
    epochs=50, 
    imgsz=640, 
    augment=True,  # Enable augmentation
    **augmentation_params
)
