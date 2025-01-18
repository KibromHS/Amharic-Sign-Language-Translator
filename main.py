import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
from model import ASLModel

# Load labels dynamically from dataset folder names (Amharic folder names supported)
def load_labels(dataset_path="dataset"):
    """Load labels dynamically from dataset folder names."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path '{dataset_path}' not found.")
    return sorted(os.listdir(dataset_path))

# Load the trained model
def load_model(model_path="model.pth", num_classes=10):
    """Load the trained model."""
    model = ASLModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Preprocess hand landmarks for the model
def preprocess(hand_landmarks):
    """Preprocess hand landmarks for model input."""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])  # Extract x, y, z coordinates

    # Normalize landmarks to zero mean and unit variance
    landmarks = np.array(landmarks, dtype=np.float32)
    landmarks = (landmarks - np.mean(landmarks)) / (np.std(landmarks) + 1e-6)

    # Add batch dimension
    return torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 63)

# Render Amharic text on a frame using Pillow
def draw_text_amharic(frame, text, position, font_path="NotoSansEthiopic-Regular.ttf", font_size=32, color=(0, 255, 0)):
    """Draw Amharic text on a frame."""
    # Convert the OpenCV frame to a PIL Image
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Load the font
    font = ImageFont.truetype(font_path, font_size)

    # Draw the text
    draw.text(position, text, font=font, fill=color)

    # Convert the PIL Image back to an OpenCV frame
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# Perform real-time hand gesture detection
def detect_hands(model, labels, font_path="NotoSansEthiopic-Regular.ttf"):
    """Capture video and predict hand signs."""
    cap = cv2.VideoCapture(0)  # Open the webcam
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    drawing_utils = mp.solutions.drawing_utils

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a natural selfie view
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks
        result = hands.process(frame_rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Preprocess landmarks for the model
                input_tensor = preprocess(hand_landmarks)

                # Predict the label
                with torch.no_grad():
                    output = model(input_tensor)
                    predicted_label_idx = torch.argmax(output, dim=1).item()
                    label_name = labels[predicted_label_idx]

                # Draw landmarks and label on the frame
                drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                frame = draw_text_amharic(frame, label_name, (10, 50), font_path)

        # Display the frame
        cv2.imshow("Hand Sign Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    dataset_path = "dataset"  # Path to dataset folders
    model_path = "model.pth"  # Path to the saved model
    font_path = "NotoSansEthiopic-Regular.ttf"  # Path to the Amharic font file

    # Load labels and model
    labels = load_labels(dataset_path)
    model = load_model(model_path, num_classes=len(labels))

    # Start real-time detection
    detect_hands(model, labels, font_path)
