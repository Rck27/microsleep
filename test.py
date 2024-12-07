import cv2
import time
import numpy as np
import mediapipe as mp
import pygame
import os
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates


def get_mediapipe_app(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,

):
    """Initialize and return Mediapipe FaceMesh Solution Graph object"""
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    return face_mesh


def distance(point_1, point_2):
    """Calculate l2-norm between two points"""
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist


def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    """
    Calculate Eye Aspect Ratio for one eye.

    Args:
        landmarks: (list) Detected landmarks list
        refer_idxs: (list) Index positions of the chosen landmarks
                            in order P1, P2, P3, P4, P5, P6
        frame_width: (int) Width of captured frame
        frame_height: (int) Height of captured frame

    Returns:
        ear: (float) Eye aspect ratio
    """
    try:
        # Compute the euclidean distance between the horizontal
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)

        # Eye landmark (x, y)-coordinates
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        # Compute the eye aspect ratio
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

    except:
        ear = 0.0
        coords_points = None

    return ear, coords_points


def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    # Calculate Eye aspect ratio
    left_ear, left_lm_coordinates = get_ear(landmarks, left_eye_idxs, image_w, image_h)
    right_ear, right_lm_coordinates = get_ear(landmarks, right_eye_idxs, image_w, image_h)
    Avg_EAR = (left_ear + right_ear) / 2.0

    return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)


def plot_eye_landmarks(frame, left_lm_coordinates, right_lm_coordinates, color):
    for lm_coordinates in [left_lm_coordinates, right_lm_coordinates]:
        if lm_coordinates:
            for coord in lm_coordinates:
                cv2.circle(frame, coord, 2, color, -1)

    frame = cv2.flip(frame, 1)
    return frame


def plot_text(image, text, origin, color, font=cv2.FONT_HERSHEY_SIMPLEX, fntScale=0.8, thickness=2):
    image = cv2.putText(image, text, origin, font, fntScale, color, thickness)
    return image


class DrowsinessDetector:
    def __init__(self, wait_time=1.0, ear_thresh=0.18, alarm_path="wake_up.wav", firestore_cred_path=None):
        """
        Initialize the drowsiness detection system.

        Args:
            wait_time (float): Seconds to wait before sounding alarm
            ear_thresh (float): Eye Aspect Ratio threshold for drowsiness
            alarm_path (str): Path to the alarm sound file
            firestore_cred_path (str): Path to Firebase credentials JSON file
        """
        # Left and right eye chosen landmarks.
        self.eye_idxs = {
            "left": [362, 385, 387, 263, 373, 380],
            "right": [33, 160, 158, 133, 153, 144],
        }

        # Used for coloring landmark points.
        # Its value depends on the current EAR value.
        self.RED = (0, 0, 255)  # BGR
        self.GREEN = (0, 255, 0)  # BGR

        # Initializing Mediapipe FaceMesh solution pipeline
        self.facemesh_model = get_mediapipe_app()

        # Thresholds and configuration
        self.wait_time = wait_time
        self.ear_thresh = ear_thresh

        # For tracking counters and sharing states
        self.state_tracker = {
            "start_time": time.perf_counter(),
            "DROWSY_TIME": 0.0,  # Holds the amount of time passed with EAR < EAR_THRESH
            "COLOR": self.GREEN,
            "play_alarm": False,
            "microsleep_logged": False  # New flag to prevent multiple logs
        }

        # Initialize pygame mixer for sound
        pygame.mixer.init()
        self.alarm_sound = pygame.mixer.Sound(alarm_path) if os.path.exists(alarm_path) else None

        # Text position
        self.EAR_txt_pos = (10, 30)

        # Initialize Firestore if credentials path is provided
        self.db = None
        if firestore_cred_path and os.path.exists(firestore_cred_path):
            try:
                cred = credentials.Certificate(firestore_cred_path)
                firebase_admin.initialize_app(cred)
                self.db = firestore.client()
                print("Firestore initialized successfully")
            except Exception as e:
                print(f"Error initializing Firestore: {e}")

    def log_microsleep(self):
        """
        Log microsleep event to Firestore only once per detection
        """
        if self.db is not None and not self.state_tracker["microsleep_logged"]:
            try:
                # Get current date and time
                current_time = datetime.now()

                # Format date as dd/mm/yy
                formatted_date = current_time.strftime("%d/%m/%y")

                # Format time as HH:MM:SS
                formatted_time = current_time.strftime("%H:%M:%S")

                # Combine date and time
                waktu = f"{formatted_date} {formatted_time}"

                # Add document to Firestore
                self.db.collection('microsleep_events').add({
                    'waktu': waktu,
                    'timestamp': firestore.SERVER_TIMESTAMP
                })
                print(f"Microsleep event logged: {waktu}")

                # Set the flag to prevent multiple logs
                self.state_tracker["microsleep_logged"] = True
            except Exception as e:
                print(f"Error logging microsleep to Firestore: {e}")

    def process_frame(self, frame):
        """
        Process a single frame for drowsiness detection.

        Args:
            frame (np.array): Input frame matrix.

        Returns:
            Processed frame with annotations
        """
        # Create a writable copy of the frame
        frame_copy = frame.copy()
        frame_copy.flags.writeable = True

        frame_h, frame_w, _ = frame_copy.shape

        DROWSY_TIME_txt_pos = (10, int(frame_h // 2 * 1.7))
        ALM_txt_pos = (10, int(frame_h // 2 * 1.85))

        results = self.facemesh_model.process(frame_copy)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            EAR, coordinates = calculate_avg_ear(
                landmarks,
                self.eye_idxs["left"],
                self.eye_idxs["right"],
                frame_w,
                frame_h
            )
            frame_copy = plot_eye_landmarks(frame_copy, coordinates[0], coordinates[1], self.state_tracker["COLOR"])

            if EAR < self.ear_thresh:
                # Increase DROWSY_TIME to track the time period with EAR less than the threshold
                # and reset the start_time for the next iteration.
                end_time = time.perf_counter()

                self.state_tracker["DROWSY_TIME"] += end_time - self.state_tracker["start_time"]
                self.state_tracker["start_time"] = end_time
                self.state_tracker["COLOR"] = self.RED

                if self.state_tracker["DROWSY_TIME"] >= self.wait_time:
                    self.state_tracker["play_alarm"] = True
                    frame_copy = plot_text(frame_copy, "WAKE UP! WAKE UP", ALM_txt_pos, self.state_tracker["COLOR"])

                    # Play alarm sound if available
                    if self.alarm_sound and not pygame.mixer.get_busy():
                        self.alarm_sound.play()

                    # Log microsleep event to Firestore
                    self.log_microsleep()

            else:
                # Reset tracking when eyes are open
                self.state_tracker["start_time"] = time.perf_counter()
                self.state_tracker["DROWSY_TIME"] = 0.0
                self.state_tracker["COLOR"] = self.GREEN
                self.state_tracker["play_alarm"] = False

                # Reset the microsleep logged flag when eyes are open
                self.state_tracker["microsleep_logged"] = False

            # Plot EAR and Drowsy Time text
            EAR_txt = f"EAR: {round(EAR, 2)}"
            DROWSY_TIME_txt = f"DROWSY: {round(self.state_tracker['DROWSY_TIME'], 3)} Secs \n"
            print(DROWSY_TIME_txt)
            frame_copy = plot_text(frame_copy, EAR_txt, self.EAR_txt_pos, self.state_tracker["COLOR"])
            frame_copy = plot_text(frame_copy, DROWSY_TIME_txt, DROWSY_TIME_txt_pos, self.state_tracker["COLOR"])

        else:
            # Reset state if no face detected
            self.state_tracker["start_time"] = time.perf_counter()
            self.state_tracker["DROWSY_TIME"] = 0.0
            self.state_tracker["COLOR"] = self.GREEN
            self.state_tracker["play_alarm"] = False
            self.state_tracker["microsleep_logged"] = False

            # Flip the frame horizontally for a selfie-view display.
            frame_copy = cv2.flip(frame_copy, 1)

        return frame_copy


def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use default camera

    # Replace 'path/to/your/firebase-credentials.json' with the actual path to your Firebase credentials
    detector = DrowsinessDetector(
        wait_time=3.0,  # Wait 3 seconds before sounding alarm
        ear_thresh=0.18,  # Eye Aspect Ratio threshold
        alarm_path="wake_up.wav",  # Path to alarm sound file
        firestore_cred_path='microsleep.json'  # Path to Firebase credentials
    )

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Process the frame
        processed_frame = detector.process_frame(frame)

        # Display the processed frame
        # cv2.imshow('Drowsiness Detection', processed_frame)


        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()