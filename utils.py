import numpy as np

def preprocess(frame, hand_landmarks, normalize=True):
    """
    Preprocess hand landmarks for model input.
    
    Args:
        frame (np.ndarray): The input image frame (used for context if needed).
        hand_landmarks (object): MediaPipe hand landmarks object.
        normalize (bool): Whether to normalize landmarks to zero mean and unit variance.
    
    Returns:
        np.ndarray: Preprocessed landmarks with shape (1, 63).
    """
    if not hand_landmarks or not hasattr(hand_landmarks, "landmark"):
        raise ValueError("Invalid hand_landmarks object provided.")

    # Extract x, y, z coordinates from landmarks
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    # Convert to a NumPy array
    landmarks = np.array(landmarks, dtype=np.float32)

    # Normalize landmarks (optional)
    if normalize:
        landmarks = (landmarks - np.mean(landmarks)) / (np.std(landmarks) + 1e-6)  # Avoid division by zero

    # Add batch dimension
    return np.expand_dims(landmarks, axis=0)  # Shape: (1, 63)
