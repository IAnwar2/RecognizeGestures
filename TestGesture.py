import cv2
import numpy as np
from tensorflow.keras.models import load_model
from HandGesture import FCNN
import mediapipe as mp
import time

video_capture = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configure MediaPipe Hands
hands = mp_hands.Hands(static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

# Load model for hand gesture recognition
model = load_model("Model/hand_gesture_model.keras")

process_this_frame = True
last_prediction_time = 0  # Last time a prediction was made
prediction_interval = 0.5  # Interval in seconds between predictions

while True:
    ret, frame = video_capture.read() # Get a frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    result = hands.process(rgb_frame)

    # Process every other frame of video to save time
    if process_this_frame and result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            current_time = time.time()
            if current_time - last_prediction_time >= prediction_interval:

                input_features = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()
                # Reshape the input to (1, 63)
                input_features = input_features.reshape(1, -1)

                # Predict the gesture
                prediction = model(input_features)
                predicted_class = np.argmax(prediction) 
                confidence = np.max(prediction)

                gesture_labels = ["Blank", "Thumbs Down", "Thumbs Up"]  
                gesture = gesture_labels[predicted_class]

                # Display the predicted gesture and confidence
                cv2.putText(frame, f"{gesture} ({confidence:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    process_this_frame = not process_this_frame

    # Display the resulting image
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): # q to quit
        break

video_capture.release()
cv2.destroyAllWindows()