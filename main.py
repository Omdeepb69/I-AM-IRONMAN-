import cv2
import mediapipe as mp
import numpy as np
import dlib
import time
import os
import urllib.request
import bz2
import shutil
from ursina import *
import math

class IronManARSuit:
    def __init__(self):
        # Initialize Ursina
        self.app = Ursina(borderless=False)
        window.exit_button.visible = True
        window.fps_counter.enabled = True
        
        # Create a camera
        self.camera = EditorCamera()
        
        # Settings
        self.show_gauntlet = False
        self.show_helmet = False
        self.last_gesture_time = 0
        self.gesture_cooldown = 1.0  # 1 second cooldown between gestures
        
        # Initialize MediaPipe for hand tracking
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Initialize dlib for face detection
        self.face_detector = dlib.get_frontal_face_detector()
        
        # Check and download shape predictor if needed
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(predictor_path):
            self.download_shape_predictor()
            
        self.shape_predictor = dlib.shape_predictor(predictor_path)
        
        # Create 3D models
        self.create_models()
        
        # Start camera capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("ERROR: Could not open camera.")
            exit()
        
        # Set up audio effects
        try:
            from ursina.audio import Audio
            self.power_up_sound = Audio('powerup', autoplay=False, loop=False)
            self.power_down_sound = Audio('powerdown', autoplay=False, loop=False)
            self.audio_enabled = True
        except:
            print("Audio features not available.")
            self.audio_enabled = False
            
        # Start the main update loop
        self.setup_update_function()
    
    def download_shape_predictor(self):
        """Download and extract the shape predictor file if it doesn't exist"""
        print("Shape predictor file not found. Downloading...")
        
        # URLs for the shape predictor
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        bz2_file = "shape_predictor_68_face_landmarks.dat.bz2"
        dat_file = "shape_predictor_68_face_landmarks.dat"
        
        try:
            # Download the file
            print("Downloading from", url)
            urllib.request.urlretrieve(url, bz2_file)
            print("Download complete. Extracting...")
            
            # Extract the bz2 file
            with bz2.BZ2File(bz2_file, 'rb') as source, open(dat_file, 'wb') as dest:
                shutil.copyfileobj(source, dest)
            
            # Remove the bz2 file
            os.remove(bz2_file)
            print("Extraction complete. Ready to use.")
            
        except Exception as e:
            print(f"Error downloading or extracting shape predictor: {e}")
            print("Please download it manually from:", url)
            print("Extract it and place it in the same directory as this script.")
            exit()
    
    def create_models(self):
        """Create Iron Man suit component models"""
        # Create gauntlet
        self.gauntlet = Entity(parent=scene)
        self.gauntlet_palm = Entity(
            parent=self.gauntlet,
            model='sphere',
            color=color.rgb(255, 50, 50),
            scale=(0.3, 0.1, 0.3),
            position=(0, 0, 0)
        )
        
        # Create the main part of the gauntlet
        self.gauntlet_base = Entity(
            parent=self.gauntlet,
            model='cube',
            color=color.rgb(200, 0, 0),
            scale=(0.25, 0.05, 0.4),
            position=(0, 0, -0.2)
        )
        
        # Create the fingers
        finger_positions = [
            (0.1, 0, -0.25), (-0.1, 0, -0.25),  # thumb and pinky
            (0.05, 0, -0.35), (0, 0, -0.4), (-0.05, 0, -0.35)  # other fingers
        ]
        
        for pos in finger_positions:
            Entity(
                parent=self.gauntlet,
                model='cube',
                color=color.rgb(180, 0, 0),
                scale=(0.03, 0.04, 0.2),
                position=pos
            )
        
        # Create repulsor
        self.repulsor = Entity(
            parent=self.gauntlet_palm,
            model='sphere',
            color=color.rgb(100, 200, 255),
            scale=(0.15, 0.05, 0.15),
            position=(0, -0.05, 0)
        )
        
        # Add metallic details
        for i in range(5):
            angle = i * (2 * math.pi / 5)
            x = 0.15 * math.cos(angle)
            z = 0.15 * math.sin(angle)
            Entity(
                parent=self.gauntlet_base,
                model='sphere',
                color=color.rgb(150, 150, 150),
                scale=(0.03, 0.02, 0.03),
                position=(x, 0.03, z-0.2)
            )
        
        # Create helmet
        self.helmet = Entity(parent=scene)
        
        # Face plate
        self.face_plate = Entity(
            parent=self.helmet,
            model='sphere',
            texture='white_cube',
            color=color.rgb(200, 0, 0),
            scale=(0.35, 0.5, 0.3)
        )
        
        # Eye slits
        self.left_eye = Entity(
            parent=self.helmet,
            model='cube',
            color=color.rgb(100, 200, 255),
            scale=(0.1, 0.03, 0.05),
            position=(-0.1, 0.1, -0.25)
        )
        
        self.right_eye = Entity(
            parent=self.helmet,
            model='cube',
            color=color.rgb(100, 200, 255),
            scale=(0.1, 0.03, 0.05),
            position=(0.1, 0.1, -0.25)
        )
        
        # Hide the models initially
        self.gauntlet.enabled = False
        self.helmet.enabled = False
    
    def setup_update_function(self):
        """Set up the main update function"""
        def update():
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    return
                
                # Flip the frame horizontally for a later selfie-view display
                frame = cv2.flip(frame, 1)
                
                # Convert the BGR image to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the image for hand tracking
                hand_results = self.hands.process(rgb_frame)
                
                # Process the image for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_rects = self.face_detector(gray, 0)
                
                # Update gauntlet position based on hand landmarks
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        # Draw hand landmarks for visualization
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        
                        # Get the position of the wrist
                        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                        
                        # Convert normalized coordinates to pixel coordinates
                        h, w, _ = frame.shape
                        wrist_x = int(wrist.x * w)
                        wrist_y = int(wrist.y * h)
                        
                        # Position the gauntlet if it's enabled
                        if self.gauntlet.enabled:
                            # Normalize coordinates for Ursina's coordinate system
                            normalized_x = (wrist_x / w - 0.5) * 2
                            normalized_y = -(wrist_y / h - 0.5) * 2
                            
                            self.gauntlet.position = (normalized_x, normalized_y, 0)
                            
                            # Get thumb and index finger for gestures
                            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                            index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                            middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                            
                            # Check for fist gesture (when all fingertips are close to the palm)
                            palm = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                            
                            # Calculate distances
                            thumb_to_palm = self.distance_3d(thumb_tip, palm)
                            index_to_palm = self.distance_3d(index_finger_tip, palm)
                            middle_to_palm = self.distance_3d(middle_finger_tip, palm)
                            
                            # Check for peace sign (index and middle up)
                            index_to_middle = self.distance_3d(index_finger_tip, middle_finger_tip)
                            
                            # Detect fist gesture (toggle gauntlet)
                            if thumb_to_palm < 0.1 and index_to_palm < 0.1 and middle_to_palm < 0.1:
                                if time.time() - self.last_gesture_time > self.gesture_cooldown:
                                    self.show_gauntlet = not self.show_gauntlet
                                    self.last_gesture_time = time.time()
                                    if self.audio_enabled:
                                        if self.show_gauntlet:
                                            self.power_up_sound.play()
                                        else:
                                            self.power_down_sound.play()
                            
                            # Detect peace sign gesture (toggle helmet)
                            if thumb_to_palm < 0.1 and index_to_palm > 0.15 and middle_to_palm > 0.15 and index_to_middle < 0.1:
                                if time.time() - self.last_gesture_time > self.gesture_cooldown:
                                    self.show_helmet = not self.show_helmet
                                    self.last_gesture_time = time.time()
                                    if self.audio_enabled:
                                        if self.show_helmet:
                                            self.power_up_sound.play()
                                        else:
                                            self.power_down_sound.play()
                
                # Handle face detection and helmet positioning
                if len(face_rects) > 0:
                    for rect in face_rects:
                        # Draw rectangle around the face
                        x1 = rect.left()
                        y1 = rect.top()
                        x2 = rect.right()
                        y2 = rect.bottom()
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Get facial landmarks
                        shape = self.shape_predictor(gray, rect)
                        
                        # Calculate face center
                        face_center_x = (x1 + x2) / 2
                        face_center_y = (y1 + y2) / 2
                        
                        # Normalize for Ursina coordinates
                        normalized_x = (face_center_x / w - 0.5) * 2
                        normalized_y = -(face_center_y / h - 0.5) * 2
                        
                        # Position the helmet if it's enabled
                        if self.helmet.enabled:
                            self.helmet.position = (normalized_x, normalized_y, 0)
                            
                            # Scale the helmet based on face size
                            face_width = (x2 - x1) / w
                            self.helmet.scale = (face_width * 2, face_width * 2, face_width * 2)
                
                # Enable or disable models based on toggle state
                self.gauntlet.enabled = self.show_gauntlet
                self.helmet.enabled = self.show_helmet
                
                # Display the processed image
                cv2.imshow('JARVIS AR System', frame)
                if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
                    self.cleanup()
        
        # Set the Ursina update function
        self.app.run_function_as_update = update
    
    def distance_3d(self, a, b):
        """Calculate 3D distance between two landmarks"""
        return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        application.quit()

def main():
    print("Starting JARVIS: Just Another Really Very Intelligent System for AR Suits")
    print("Loading Iron Man AR experience...")
    print("Initializing hand tracking and face detection...")
    
    # Create and run the application
    iron_man_ar = IronManARSuit()

if __name__ == "__main__":
    main()