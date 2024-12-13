import cv2
import time
import numpy as np
import mediapipe as mp
import pygame
import os
import json
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates



try:
    with open('microsleep.json', 'r') as cred_file:
        cred_data = json.load(cred_file)
        print("Firebase Credentials:")
        print(f"Project ID: {cred_data.get('project_id')}")
        print(f"Client Email: {cred_data.get('client_email')}")
except Exception as e:
    print(f"Error reading credentials: {e}")

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


def get_mor(landmarks, refer_idxs, frame_width, frame_height):
    """
    Calculate Mouth Open Ratio.

    Args:
        landmarks: (list) Detected landmarks list
        refer_idxs: (list) Index positions of the chosen landmarks
        frame_width: (int) Width of captured frame
        frame_height: (int) Height of captured frame

    Returns:
        mor: (float) Mouth Open Ratio
        coords_points: (list) Coordinates of landmark points
    """
    try:
        # Compute the euclidean distance between mouth landmarks
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)

        # Vertical mouth distance
        upper_lip_center = coords_points[0]
        lower_lip_center = coords_points[1]
        mouth_height = distance(upper_lip_center, lower_lip_center)

        # Horizontal mouth width (mouth corners)
        left_corner = coords_points[2]
        right_corner = coords_points[3]
        mouth_width = distance(left_corner, right_corner)

        # Compute the mouth open ratio
        mor = mouth_height / mouth_width if mouth_width > 0 else 0.0

    except:
        mor = 0.0
        coords_points = None

    return mor, coords_points


def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    # Calculate Eye aspect ratio
    left_ear, left_lm_coordinates = get_ear(landmarks, left_eye_idxs, image_w, image_h)
    right_ear, right_lm_coordinates = get_ear(landmarks, right_eye_idxs, image_w, image_h)
    Avg_EAR = (left_ear + right_ear) / 2.0

    return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)


def plot_landmarks(frame, landmarks_list, color):
    """
    Plot landmarks on the frame

    Args:
        frame: Input frame
        landmarks_list: List of landmark coordinates
        color: Color to plot landmarks

    Returns:
        Modified frame
    """
    if landmarks_list:
        for coord in landmarks_list:
            cv2.circle(frame, coord, 2, color, -1)

    frame = cv2.flip(frame, 1)
    return frame


def plot_text(image, text, origin, color, font=cv2.FONT_HERSHEY_SIMPLEX, fntScale=0.8, thickness=2):
    image = cv2.putText(image, text, origin, font, fntScale, color, thickness)
    return image


class DrowsinessDetector:
    def __init__(self,
                 wait_time=1.0,
                 ear_thresh=0.18,
                 mor_thresh=0.3,  # New parameter for mouth open ratio threshold
                 alarm_path="wake_up.wav",
                 firestore_cred_path=None):
        """
        Initialize the drowsiness detection system.

        Args:
            wait_time (float): Seconds to wait before sounding alarm
            ear_thresh (float): Eye Aspect Ratio threshold for drowsiness
            mor_thresh (float): Mouth Open Ratio threshold
            alarm_path (str): Path to the alarm sound file
            firestore_cred_path (str): Path to Firebase credentials JSON file
        """

        #initiate gpio 2 for output and set to 0
        os.system("gpio mode 2 out \ gpio write 2 0")

        # Landmarks indices
        self.eye_idxs = {
            "left": [362, 385, 387, 263, 373, 380],
            "right": [33, 160, 158, 133, 153, 144],
        }

        # Mouth landmarks indices
        # Upper lip center, lower lip center, left corner, right corner
        self.mouth_idxs = [13, 14, 78, 308]

        # Used for coloring landmark points.
        self.RED = (0, 0, 255)  # BGR
        self.GREEN = (0, 255, 0)  # BGR
        self.BLUE = (255, 0, 0)  # BGR

        # Initializing Mediapipe FaceMesh solution pipeline
        self.facemesh_model = get_mediapipe_app()

        # Thresholds and configuration
        self.wait_time = wait_time
        self.ear_thresh = ear_thresh
        self.mor_thresh = mor_thresh  # New threshold for mouth open ratio

        # For tracking counters and sharing states
        self.state_tracker = {
            "start_time": time.perf_counter(),
            "DROWSY_TIME": 0.0,  # Holds the amount of time passed with EAR < EAR_THRESH
            "COLOR": self.GREEN,
            "play_alarm": False,
            "microsleep_logged": False  # Flag to prevent multiple logs
        }

        # Initialize pygame mixer for sound
        pygame.mixer.init()
        self.alarm_sound = pygame.mixer.Sound(alarm_path) if os.path.exists(alarm_path) else None

        # Text position
        self.EAR_txt_pos = (10, 30)
        self.MOR_txt_pos = (10, 60)  # New position for Mouth Open Ratio text

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
        Log microsleep event to Firestore with improved error handling
        """
        if self.db is not None and not self.state_tracker["microsleep_logged"]:
            try:
                # Verify Firebase app is initialized
                if not firebase_admin._apps:
                    print("Firebase app not initialized. Reinitializing...")
                    firestore_cred_path = 'microsleep.json'
                    cred = credentials.Certificate(firestore_cred_path)
                    firebase_admin.initialize_app(cred)
                    self.db = firestore.client()

                # Get current date and time
                current_time = datetime.now()

                # Format date as dd/mm/yy
                formatted_date = current_time.strftime("%d/%m/%y")

                # Format time as HH:MM:SS
                formatted_time = current_time.strftime("%H:%M:%S")

                # Combine date and time
                waktu = f"{formatted_date} {formatted_time}"

                # Add document to Firestore with extended timeout
                microsleep_ref = self.db.collection('microsleep_events')
                microsleep_ref.add({
                    'waktu': waktu,
                    'timestamp': firestore.SERVER_TIMESTAMP
                }, timeout=30.0)  # Extend timeout to 30 seconds

                print(f"Microsleep event logged: {waktu}")

                # Set the flag to prevent multiple logs
                self.state_tracker["microsleep_logged"] = True

            except firebase_admin.exceptions.FirebaseError as firebase_err:
                print(f"Firebase Error: {firebase_err}")
                # Attempt to reinitialize Firebase
                try:
                    firebase_admin.delete_app(firebase_admin.get_app())
                except:
                    pass

            except Exception as e:
                print(f"Unexpected error logging microsleep to Firestore: {e}")
                # Log the full traceback for debugging
                import traceback
                traceback.print_exc()

    def process_frame(self, frame):
        # Create a writable copy of the frame
        frame_copy = frame.copy()
        frame_copy.flags.writeable = True

        frame_h, frame_w, _ = frame_copy.shape

        DROWSY_TIME_txt_pos = (10, int(frame_h // 2 * 1.7))
        ALM_txt_pos = (10, int(frame_h // 2 * 1.85))

        results = self.facemesh_model.process(frame_copy)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Calculate Eye Aspect Ratio (EAR)
            EAR, eye_coordinates = calculate_avg_ear(
                landmarks,
                self.eye_idxs["left"],
                self.eye_idxs["right"],
                frame_w,
                frame_h
            )

            # Calculate Mouth Open Ratio (MOR)
            MOR, mouth_coordinates = get_mor(
                landmarks,
                self.mouth_idxs,
                frame_w,
                frame_h
            )

            # Plot eye landmarks
            frame_copy = plot_landmarks(frame_copy,
                                        eye_coordinates[0] + eye_coordinates[1],
                                        self.state_tracker["COLOR"])

            # Plot mouth landmarks
            frame_copy = plot_landmarks(frame_copy,
                                        mouth_coordinates,
                                        self.BLUE)

            # Drowsiness detection logic
            drowsy_condition = (EAR < self.ear_thresh) or (MOR > self.mor_thresh)

            if drowsy_condition:
                # Increase DROWSY_TIME to track the time period
                # and reset the start_time for the next iteration.
                end_time = time.perf_counter()
                isAlarmOn = False
                self.state_tracker["DROWSY_TIME"] += end_time - self.state_tracker["start_time"]
                self.state_tracker["start_time"] = end_time
                self.state_tracker["COLOR"] = self.RED

                if self.state_tracker["DROWSY_TIME"] >= self.wait_time:
                    self.state_tracker["play_alarm"] = True
                    frame_copy = plot_text(frame_copy, "WAKE UP! WAKE UP", ALM_txt_pos, self.state_tracker["COLOR"])
                    if not isAlarmOn:
                        os.system("gpio write 2 1")
                        isAlarmOn = not isAlarmOn
                        print(f"alarm f is {isAlarmOn}\n")
                    else:
                        os.system("gpio write 2 0")
                        print(f"alarm {isAlarmOn}\n")
                        isAlarmOn = not isAlarmOn
                              
                    # Play alarm sound if available
                    if self.alarm_sound and not pygame.mixer.get_busy():
                        self.alarm_sound.play()

                    # Log microsleep event to Firestore
                    self.log_microsleep()

            else:
                # Reset tracking when face looks normal
                self.state_tracker["start_time"] = time.perf_counter()
                self.state_tracker["DROWSY_TIME"] = 0.0
                self.state_tracker["COLOR"] = self.GREEN
                self.state_tracker["play_alarm"] = False
                self.state_tracker["microsleep_logged"] = False
                os.system("gpio write 2 0")
                isAlarmOn = 0
                print("Alarm is off")

            # Plot EAR and MOR texts
            EAR_txt = f"EAR: {round(EAR, 2)}"
            MOR_txt = f"MOR: {round(MOR, 2)}"
            DROWSY_TIME_txt = f"DROWSY: {round(self.state_tracker['DROWSY_TIME'], 3)} Secs"

            frame_copy = plot_text(frame_copy, EAR_txt, self.EAR_txt_pos, self.state_tracker["COLOR"])
            frame_copy = plot_text(frame_copy, MOR_txt, self.MOR_txt_pos, self.BLUE)
            frame_copy = plot_text(frame_copy, DROWSY_TIME_txt, DROWSY_TIME_txt_pos, self.state_tracker["COLOR"])

        else:
            # Reset state if no face detected
            self.state_tracker["start_time"] = time.perf_counter()
            self.state_tracker["DROWSY_TIME"] = 0.0
            self.state_tracker["COLOR"] = self.GREEN
            self.state_tracker["play_alarm"] = False
            self.state_tracker["microsleep_logged"] = False
            os.system("gpio write 2 0")
            isAlarmOn = 0
            # Flip the frame horizontally for a selfie-view display.
            frame_copy = cv2.flip(frame_copy, 1)

        return frame_copy


def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use default camera

    # Initialize detector with updated parameters
    detector = DrowsinessDetector(
        wait_time=3.0,  # Wait 3 seconds before sounding alarm
        ear_thresh=0.18,  # Eye Aspect Ratio threshold
        mor_thresh=0.4,  # Mouth Open Ratio threshold
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
        cv2.imshow('Drowsiness Detection', processed_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()