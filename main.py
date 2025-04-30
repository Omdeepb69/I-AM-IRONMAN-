import cv2
import mediapipe as mp
import numpy as np
import math
from ursina import *
import time
import threading

class IronManAR:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize hand tracking
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize face tracking
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Tracking data
        self.hand_landmarks = None
        self.face_landmarks = None
        self.hand_dimensions = {'width': 0, 'length': 0}
        self.face_dimensions = {'width': 0, 'height': 0}
        
        # Animation states
        self.gauntlet_deployed = False
        self.helmet_deployed = False
        self.deployment_progress = 0
        self.last_gesture = None
        
        # Threading setup for concurrent video capture and 3D rendering
        self.is_running = True
        self.frame = None
        self.frame_ready = False
        
        # Start the webcam thread
        self.webcam_thread = threading.Thread(target=self.capture_frames)
        self.webcam_thread.daemon = True
        self.webcam_thread.start()
        
        # Initialize Ursina after setting up tracking
        self.setup_ursina_engine()
    
    def capture_frames(self):
        """Continuously capture and process webcam frames in a separate thread"""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Flip the frame horizontally for a mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand landmarks
            hand_results = self.hands.process(rgb_frame)
            if hand_results.multi_hand_landmarks:
                self.hand_landmarks = hand_results.multi_hand_landmarks[0]
                self.calculate_hand_dimensions(self.hand_landmarks, rgb_frame.shape)
                self.detect_hand_gesture(self.hand_landmarks)
                
                # Draw hand landmarks for debugging
                self.mp_drawing.draw_landmarks(
                    frame, self.hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            else:
                self.hand_landmarks = None
            
            # Process face landmarks
            face_results = self.face_mesh.process(rgb_frame)
            if face_results.multi_face_landmarks:
                self.face_landmarks = face_results.multi_face_landmarks[0]
                self.calculate_face_dimensions(self.face_landmarks, rgb_frame.shape)
                
                # Draw face landmarks for debugging (sparse version for clarity)
                connections = mp.solutions.face_mesh_connections.FACEMESH_CONTOURS
                self.mp_drawing.draw_landmarks(
                    frame, self.face_landmarks, connections,
                    landmark_drawing_spec=None)
            else:
                self.face_landmarks = None
            
            # Update animation state based on gestures
            self.update_animation_states()
            
            # Make the processed frame available to the main thread
            self.frame = frame
            self.frame_ready = True
            
    def calculate_hand_dimensions(self, landmarks, frame_shape):
        """Calculate key hand dimensions from landmarks"""
        if not landmarks:
            return
            
        h, w, _ = frame_shape
        points = []
        for landmark in landmarks.landmark:
            points.append((int(landmark.x * w), int(landmark.y * h)))
        
        # Get key points for sizing
        wrist = points[0]
        middle_finger_tip = points[12]
        thumb_tip = points[4]
        pinky_tip = points[20]
        
        # Calculate dimensions
        self.hand_dimensions['length'] = self.distance(wrist, middle_finger_tip)
        self.hand_dimensions['width'] = self.distance(thumb_tip, pinky_tip)
        
    def calculate_face_dimensions(self, landmarks, frame_shape):
        """Calculate key face dimensions from landmarks"""
        if not landmarks:
            return
            
        h, w, _ = frame_shape
        points = []
        for landmark in landmarks.landmark:
            points.append((int(landmark.x * w), int(landmark.y * h)))
        
        # Use specific face mesh indices for dimensions
        # These indices represent the leftmost and rightmost points of the face
        left_face = points[234]  # Left cheek landmark
        right_face = points[454]  # Right cheek landmark
        top_face = points[10]    # Top of forehead
        bottom_face = points[152]  # Bottom of chin
        
        self.face_dimensions['width'] = self.distance(left_face, right_face)
        self.face_dimensions['height'] = self.distance(top_face, bottom_face)
    
    def distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def detect_hand_gesture(self, landmarks):
        """Detect gestures to trigger suit deployment"""
        if not landmarks:
            return
            
        # Get finger states (extended/closed)
        finger_states = self.get_finger_states(landmarks)
        
        # Detect "fist" gesture to deploy gauntlet
        if finger_states == [0, 0, 0, 0, 0]:  # All fingers closed
            if self.last_gesture != "fist":
                self.last_gesture = "fist"
                self.gauntlet_deployed = not self.gauntlet_deployed
        
        # Detect "peace" gesture to deploy helmet
        elif finger_states == [0, 1, 1, 0, 0]:  # Index and middle extended
            if self.last_gesture != "peace":
                self.last_gesture = "peace"
                self.helmet_deployed = not self.helmet_deployed
        else:
            self.last_gesture = None
    
    def get_finger_states(self, landmarks):
        """Determine if each finger is extended (1) or closed (0)"""
        # Get finger positions
        positions = []
        for landmark in landmarks.landmark:
            positions.append((landmark.x, landmark.y, landmark.z))
        
        # Finger tips and bases for each finger
        fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        finger_bases = [2, 5, 9, 13, 17]  # Corresponding bases
        wrist = positions[0]
        
        # Check each finger
        finger_states = []
        for tip_id, base_id in zip(fingertips, finger_bases):
            # For thumb, compare with wrist
            if tip_id == 4:
                if positions[tip_id][0] < positions[base_id][0]:  # Left hand check
                    finger_states.append(1)
                else:
                    finger_states.append(0)
            else:
                # For other fingers, compare y positions
                if positions[tip_id][1] < positions[base_id][1]:
                    finger_states.append(1)  # Extended
                else:
                    finger_states.append(0)  # Closed
        
        return finger_states
    
    def update_animation_states(self):
        """Update animation states based on detected gestures"""
        # Update deployment progress for animations
        if self.gauntlet_deployed and self.deployment_progress < 1.0:
            self.deployment_progress += 0.05
        elif not self.gauntlet_deployed and self.deployment_progress > 0:
            self.deployment_progress -= 0.05
        
        # Clamp values
        self.deployment_progress = max(0, min(1.0, self.deployment_progress))
    
    def setup_ursina_engine(self):
        """Initialize the Ursina game engine and 3D models"""
        # Initialize Ursina app
        self.app = Ursina()
        
        # Set up camera
        self.camera = EditorCamera()
        self.camera.position = (0, 0, -15)
        self.camera.rotation = (0, 0, 0)
        
        # Set up lighting
        self.light = DirectionalLight()
        self.light.position = (3, 5, -10)
        self.light.look_at(Vec3(0, 0, 0))
        
        # Create Iron Man hand model
        self.gauntlet = Entity(
            model='cube',
            color=color.red,
            scale=(2, 0.5, 1.5),
            position=(0, 0, 0),
            shader=lit_with_shadows_shader
        )
        
        # Finger components
        self.fingers = []
        for i in range(5):
            finger = Entity(
                model='cube',
                color=color.red,
                scale=(0.2, 0.8, 0.2),
                position=(0.7 - i * 0.35, 1, 0),
                shader=lit_with_shadows_shader
            )
            self.fingers.append(finger)
        
        # Palm repulsor
        self.palm_repulsor = Entity(
            model='sphere',
            color=color.cyan,
            scale=(0.3, 0.1, 0.3),
            position=(0, 0, -0.8),
            shader=lit_with_shadows_shader
        )
        
        # Create Iron Man helmet model
        self.helmet = Entity(
            model='sphere',
            color=color.red,
            scale=(2, 2.5, 2),
            position=(0, 5, 0),
            shader=lit_with_shadows_shader
        )
        
        # Face plate
        self.face_plate = Entity(
            model='cube',
            color=color.red,
            scale=(1.8, 1.2, 0.5),
            position=(0, 5, -1),
            shader=lit_with_shadows_shader
        )
        
        # Eye slits
        self.left_eye = Entity(
            model='cube',
            color=color.cyan,
            scale=(0.5, 0.2, 0.1),
            position=(-0.5, 5.3, -1.3),
            shader=lit_with_shadows_shader
        )
        
        self.right_eye = Entity(
            model='cube',
            color=color.cyan,
            scale=(0.5, 0.2, 0.1),
            position=(0.5, 5.3, -1.3),
            shader=lit_with_shadows_shader
        )
        
        # Create a video texture for background
        self.video_texture = Entity(
            model='quad',
            scale=(16, 9),
            position=(0, 0, 10),
            texture=Texture(np.zeros((720, 1280, 3), dtype=np.uint8))
        )
        
        # Set up update function
        self.app.run()
    
    def update(self):
        """Main update function for Ursina app"""
        if not self.frame_ready:
            return
            
        # Update video background texture
        if self.frame is not None:
            texture_data = cv2.resize(self.frame, (1280, 720))
            texture_data = cv2.flip(texture_data, 0)  # Flip vertically for Ursina
            self.video_texture.texture = Texture(texture_data)
            self.frame_ready = False
        
        # Update hand model position and scale based on hand tracking
        if self.hand_landmarks:
            # Position the gauntlet based on hand position
            hand_x = (self.hand_landmarks.landmark[0].x - 0.5) * 20
            hand_y = (0.5 - self.hand_landmarks.landmark[0].y) * 20
            depth = -5 - (self.hand_landmarks.landmark[0].z * 10)
            
            # Apply hand dimensions to scale
            width_scale = self.hand_dimensions['width'] / 200
            length_scale = self.hand_dimensions['length'] / 200
            
            # Update gauntlet position and scale
            self.gauntlet.position = (hand_x, hand_y, depth)
            self.gauntlet.scale = (2 * width_scale, 0.5, 1.5 * length_scale)
            
            # Animate based on deployment state
            if self.gauntlet_deployed:
                # Make the model fully visible
                self.gauntlet.color = color.rgba(200, 50, 50, 255)
                for finger in self.fingers:
                    finger.color = color.rgba(200, 50, 50, 255)
                self.palm_repulsor.color = color.rgba(0, 255, 255, 255)
            else:
                # Make the model semi-transparent
                self.gauntlet.color = color.rgba(200, 50, 50, 50)
                for finger in self.fingers:
                    finger.color = color.rgba(200, 50, 50, 50)
                self.palm_repulsor.color = color.rgba(0, 255, 255, 50)
            
            # Update fingers
            for i, finger in enumerate(self.fingers):
                # Get finger base and tip landmarks
                if i == 0:  # Thumb
                    base_id = 2
                    tip_id = 4
                else:
                    base_id = 5 + (i-1)*4
                    tip_id = 8 + (i-1)*4
                
                base_x = (self.hand_landmarks.landmark[base_id].x - 0.5) * 20
                base_y = (0.5 - self.hand_landmarks.landmark[base_id].y) * 20
                tip_x = (self.hand_landmarks.landmark[tip_id].x - 0.5) * 20
                tip_y = (0.5 - self.hand_landmarks.landmark[tip_id].y) * 20
                
                # Position at the middle of the finger
                finger_x = (base_x + tip_x) / 2
                finger_y = (base_y + tip_y) / 2
                
                # Calculate finger length and angle
                dx = tip_x - base_x
                dy = tip_y - base_y
                finger_length = math.sqrt(dx**2 + dy**2)
                finger_angle = math.degrees(math.atan2(dy, dx))
                
                # Update position and rotation
                finger.position = (finger_x, finger_y, depth)
                finger.rotation_z = finger_angle - 90
                finger.scale = (0.2, finger_length, 0.2)
            
            # Update palm repulsor
            self.palm_repulsor.position = (hand_x, hand_y, depth - 0.8)
            
            # Animate repulsor glow based on deployment
            if self.gauntlet_deployed:
                self.palm_repulsor.scale = (
                    0.3 + math.sin(time.time() * 10) * 0.05,
                    0.1,
                    0.3 + math.sin(time.time() * 10) * 0.05
                )
        
        # Update helmet position and scale based on face tracking
        if self.face_landmarks:
            # Calculate center of the face
            nose_tip = self.face_landmarks.landmark[4]
            face_x = (nose_tip.x - 0.5) * 20
            face_y = (0.5 - nose_tip.y) * 20 + 2  # Offset to place helmet correctly
            face_z = -5 - (nose_tip.z * 10)
            
            # Scale based on face dimensions
            width_scale = self.face_dimensions['width'] / 200
            height_scale = self.face_dimensions['height'] / 200
            
            # Update helmet position and scale
            self.helmet.position = (face_x, face_y, face_z)
            self.face_plate.position = (face_x, face_y, face_z - 1)
            self.left_eye.position = (face_x - 0.5, face_y + 0.3, face_z - 1.3)
            self.right_eye.position = (face_x + 0.5, face_y + 0.3, face_z - 1.3)
            
            # Scale helmet
            self.helmet.scale = (2 * width_scale, 2.5 * height_scale, 2 * width_scale)
            self.face_plate.scale = (1.8 * width_scale, 1.2 * height_scale, 0.5)
            
            # Animate based on deployment state
            if self.helmet_deployed:
                # Make the model fully visible
                self.helmet.color = color.rgba(200, 50, 50, 255)
                self.face_plate.color = color.rgba(200, 50, 50, 255)
                self.left_eye.color = color.rgba(0, 255, 255, 255)
                self.right_eye.color = color.rgba(0, 255, 255, 255)
                
                # Animate eye glow
                glow = 0.5 + 0.5 * math.sin(time.time() * 5)
                self.left_eye.scale = (0.5, 0.2, 0.1 + glow * 0.1)
                self.right_eye.scale = (0.5, 0.2, 0.1 + glow * 0.1)
            else:
                # Make the model semi-transparent
                self.helmet.color = color.rgba(200, 50, 50, 50)
                self.face_plate.color = color.rgba(200, 50, 50, 50)
                self.left_eye.color = color.rgba(0, 255, 255, 50)
                self.right_eye.color = color.rgba(0, 255, 255, 50)
    
    def run(self):
        """Run the main application loop"""
        # Set up the update function
        def update_wrapper():
            self.update()
        
        self.app.update = update_wrapper
        
        try:
            # Run the app
            self.app.run()
        finally:
            # Clean up resources
            self.is_running = False
            self.cap.release()
            self.hands.close()
            self.face_mesh.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create and run the application
    iron_man_ar = IronManAR()
    iron_man_ar.run()
