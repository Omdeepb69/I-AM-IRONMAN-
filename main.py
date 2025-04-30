import cv2
import mediapipe as mp
import numpy as np
import math
from ursina import *
import time
import threading
from ursina.mesh_importer import *
import random

class IronManAR:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize hand tracking with improved parameters
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Initialize face tracking with improved parameters
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            refine_landmarks=True  # Enable refined landmarks for better eye/lip tracking
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
        self.chest_deployed = False
        self.deployment_progress = 0
        self.helmet_deployment_progress = 0
        self.last_gesture = None
        self.gesture_cooldown = 0  # Cooldown timer to prevent rapid toggling
        
        # Repulsor states
        self.repulsor_charging = False
        self.repulsor_charge_level = 0
        self.repulsor_blast_active = False
        self.repulsor_blast_timer = 0
        self.repulsor_particles = []
        
        # Sound effects (represented by visual indicators)
        self.sound_indicators = []
        
        # HUD elements
        self.hud_elements = []
        
        # Colors
        self.iron_man_red = color.rgb(210, 50, 40)
        self.iron_man_gold = color.rgb(255, 215, 0)
        self.repulsor_blue = color.rgb(30, 225, 255)
        
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
                
                # Draw hand landmarks for debugging (can be disabled in production)
                self.mp_drawing.draw_landmarks(
                    frame, self.hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())
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
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                        color=(0, 255, 0), thickness=1, circle_radius=1))
            else:
                self.face_landmarks = None
            
            # Update animation state based on gestures
            self.update_animation_states()
            
            # Add HUD elements to frame
            self.add_hud_to_frame(frame)
            
            # Make the processed frame available to the main thread
            self.frame = frame
            self.frame_ready = True
            
    def add_hud_to_frame(self, frame):
        """Add HUD elements to the frame for visual feedback"""
        # Add gesture recognition status
        if self.last_gesture:
            cv2.putText(frame, f"Gesture: {self.last_gesture}", (20, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add deployment status indicators
        status_y = 80
        if self.gauntlet_deployed:
            cv2.putText(frame, "Gauntlet: ACTIVE", (20, status_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Gauntlet: STANDBY", (20, status_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        status_y += 30
        
        if self.helmet_deployed:
            cv2.putText(frame, "Helmet: ACTIVE", (20, status_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Helmet: STANDBY", (20, status_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        status_y += 30
        
        if self.chest_deployed:
            cv2.putText(frame, "Chest: ACTIVE", (20, status_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Chest: STANDBY", (20, status_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add repulsor charge level indicator if charging
        if self.repulsor_charging and self.repulsor_charge_level > 0:
            charge_width = int(200 * (self.repulsor_charge_level / 100))
            cv2.rectangle(frame, (20, 170), (20 + charge_width, 190), (30, 225, 255), -1)
            cv2.rectangle(frame, (20, 170), (220, 190), (255, 255, 255), 2)
            cv2.putText(frame, f"REPULSOR: {int(self.repulsor_charge_level)}%", (30, 185), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
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
        """Detect gestures to trigger suit deployment and actions"""
        if not landmarks or self.gesture_cooldown > 0:
            return
            
        # Get finger states (extended/closed)
        finger_states = self.get_finger_states(landmarks)
        
        # Detect "fist" gesture to deploy gauntlet
        if finger_states == [0, 0, 0, 0, 0]:  # All fingers closed
            if self.last_gesture != "fist":
                self.last_gesture = "fist"
                self.gauntlet_deployed = not self.gauntlet_deployed
                self.gesture_cooldown = 30  # ~1 second at 30fps
        
        # Detect "peace" gesture to deploy helmet
        elif finger_states == [0, 1, 1, 0, 0]:  # Index and middle extended
            if self.last_gesture != "peace":
                self.last_gesture = "peace"
                self.helmet_deployed = not self.helmet_deployed
                self.gesture_cooldown = 30
        
        # Detect "thumbs up" gesture to deploy chest piece
        elif finger_states == [1, 0, 0, 0, 0]:  # Only thumb extended
            if self.last_gesture != "thumbs_up":
                self.last_gesture = "thumbs_up"
                self.chest_deployed = not self.chest_deployed
                self.gesture_cooldown = 30
        
        # Detect "repulsor charge" gesture (palm open with fingers together)
        elif finger_states == [0, 1, 1, 1, 1]:  # All fingers except thumb extended
            self.last_gesture = "repulsor_charge"
            self.repulsor_charging = True
            if self.repulsor_charge_level < 100:
                self.repulsor_charge_level += 2
        
        # Detect "repulsor fire" gesture (only index finger extended like pointing)
        elif finger_states == [0, 1, 0, 0, 0] and self.repulsor_charge_level > 30:
            if self.last_gesture != "repulsor_fire":
                self.last_gesture = "repulsor_fire"
                self.repulsor_blast_active = True
                self.repulsor_blast_timer = 15  # ~0.5 second blast
                self.gesture_cooldown = 20
                # Create repulsor blast particles
                self.create_repulsor_blast()
        else:
            self.repulsor_charging = False
            if self.repulsor_charge_level > 0:
                self.repulsor_charge_level -= 1
    
    def create_repulsor_blast(self):
        """Create particle effect for repulsor blast"""
        if not self.hand_landmarks:
            return
            
        # Reset particles list
        self.repulsor_particles = []
        
        # Get palm position
        palm_x = (self.hand_landmarks.landmark[0].x - 0.5) * 20
        palm_y = (0.5 - self.hand_landmarks.landmark[0].y) * 20
        palm_z = -5 - (self.hand_landmarks.landmark[0].z * 10)
        
        # Create particles emanating from palm
        num_particles = int(30 + (self.repulsor_charge_level / 100) * 50)
        
        for _ in range(num_particles):
            # Calculate random direction vector (mostly forward)
            direction = Vec3(
                random.uniform(-0.3, 0.3),
                random.uniform(-0.3, 0.3),
                random.uniform(-1.5, -0.8)
            )
            
            # Normalize and scale by random velocity
            velocity = direction.normalized() * random.uniform(0.3, 1.0)
            
            # Create particle
            particle = Entity(
                model='sphere',
                color=self.repulsor_blue, 
                position=Vec3(palm_x, palm_y, palm_z - 0.8),
                scale=random.uniform(0.05, 0.15),
                shader=lit_with_shadows_shader,
                always_on_top=True,
            )
            
            # Store particle with its velocity
            self.repulsor_particles.append({
                'entity': particle,
                'velocity': velocity,
                'life': random.uniform(0.5, 1.5)
            })
        
        # Reset charge level after blast
        self.repulsor_charge_level = max(0, self.repulsor_charge_level - 30)
    
    def update_repulsor_particles(self):
        """Update repulsor blast particles movement and lifespan"""
        particles_to_remove = []
        
        for particle in self.repulsor_particles:
            # Move particle
            particle['entity'].position += particle['velocity']
            
            # Decrease life
            particle['life'] -= time.dt
            
            # Fade out based on remaining life
            alpha = min(255, int(255 * particle['life']))
            particle['entity'].color = color.rgba(
                self.repulsor_blue.r, 
                self.repulsor_blue.g, 
                self.repulsor_blue.b, 
                alpha / 255
            )
            
            # Gradually decrease scale
            particle['entity'].scale *= 0.97
            
            # Mark for removal if expired
            if particle['life'] <= 0:
                particles_to_remove.append(particle)
        
        # Remove expired particles
        for particle in particles_to_remove:
            destroy(particle['entity'])
            self.repulsor_particles.remove(particle)
    
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
        # Update cooldown timer
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1
        
        # Update deployment progress for gauntlet animations
        if self.gauntlet_deployed and self.deployment_progress < 1.0:
            self.deployment_progress += 0.05
        elif not self.gauntlet_deployed and self.deployment_progress > 0:
            self.deployment_progress -= 0.05
        
        # Update deployment progress for helmet animations
        if self.helmet_deployed and self.helmet_deployment_progress < 1.0:
            self.helmet_deployment_progress += 0.05
        elif not self.helmet_deployed and self.helmet_deployment_progress > 0:
            self.helmet_deployment_progress -= 0.05
        
        # Update repulsor blast timer
        if self.repulsor_blast_timer > 0:
            self.repulsor_blast_timer -= 1
        else:
            self.repulsor_blast_active = False
        
        # Clamp values
        self.deployment_progress = max(0, min(1.0, self.deployment_progress))
        self.helmet_deployment_progress = max(0, min(1.0, self.helmet_deployment_progress))
    
    def setup_ursina_engine(self):
        """Initialize the Ursina game engine and build detailed 3D models"""
        # Initialize Ursina app
        self.app = Ursina()
        
        # Set up camera
        self.camera = EditorCamera()
        self.camera.position = (0, 0, -15)
        self.camera.rotation = (0, 0, 0)
        
        # Set up lighting
        self.main_light = DirectionalLight()
        self.main_light.position = (3, 5, -10)
        self.main_light.look_at(Vec3(0, 0, 0))
        
        # Additional point light for highlights
        self.highlight_light = PointLight()
        self.highlight_light.position = (0, 0, -8)
        self.highlight_light.color = color.rgb(255, 255, 255)
        self.highlight_light.intensity = 0.7
        
        # Create video texture for background
        self.video_texture = Entity(
            model='quad',
            scale=(16, 9),
            position=(0, 0, 10),
            texture=Texture(np.zeros((720, 1280, 3), dtype=np.uint8))
        )
        
        # Create Iron Man gauntlet components
        self.create_gauntlet_model()
        
        # Create Iron Man helmet components
        self.create_helmet_model()
        
        # Create chest piece
        self.create_chest_model()
        
        # Create HUD elements
        self.create_hud_elements()
        
        # Set up update function
        self.app.run()
    
    def create_gauntlet_model(self):
        """Create detailed Iron Man gauntlet model using mesh generation"""
        # Main hand plate (palm and back of hand)
        self.gauntlet_base = Entity(
            model=Mesh(vertices=[
                Vec3(-1, -0.3, 0.8),  # Bottom left back
                Vec3(1, -0.3, 0.8),   # Bottom right back
                Vec3(1, 0.3, 0.8),    # Top right back
                Vec3(-1, 0.3, 0.8),   # Top left back
                Vec3(-0.8, -0.4, -0.8),  # Bottom left front
                Vec3(0.8, -0.4, -0.8),   # Bottom right front
                Vec3(0.8, 0.4, -0.8),    # Top right front
                Vec3(-0.8, 0.4, -0.8),   # Top left front
            ], triangles=[
                0, 1, 2, 0, 2, 3,  # Back face
                4, 5, 6, 4, 6, 7,  # Front face
                0, 3, 7, 0, 7, 4,  # Left face
                1, 5, 6, 1, 6, 2,  # Right face
                3, 2, 6, 3, 6, 7,  # Top face
                0, 1, 5, 0, 5, 4,  # Bottom face
            ]),
            color=self.iron_man_red,
            scale=(2, 0.5, 1.5),
            position=(0, 0, 0),
            shader=lit_with_shadows_shader
        )
        
        # Wrist armor piece
        self.wrist_armor = Entity(
            model=Mesh(vertices=[
                Vec3(-1.2, -0.4, 1.2),  # Bottom left
                Vec3(1.2, -0.4, 1.2),   # Bottom right
                Vec3(1.2, 0.4, 1.2),    # Top right
                Vec3(-1.2, 0.4, 1.2),   # Top left
                Vec3(-1, -0.3, 0.8),  # Connect to hand
                Vec3(1, -0.3, 0.8),
                Vec3(1, 0.3, 0.8),
                Vec3(-1, 0.3, 0.8),
            ], triangles=[
                0, 1, 2, 0, 2, 3,  # Outer face
                4, 5, 6, 4, 6, 7,  # Inner face
                0, 3, 7, 0, 7, 4,  # Left face
                1, 5, 6, 1, 6, 2,  # Right face
                3, 2, 6, 3, 6, 7,  # Top face
                0, 1, 5, 0, 5, 4,  # Bottom face
            ]),
            color=self.iron_man_red,
            scale=(2, 0.5, 1.5),
            position=(0, 0, 0),
            shader=lit_with_shadows_shader
        )
        
        # Gold accent trim pieces
        self.wrist_trim = Entity(
            model='torus',
            color=self.iron_man_gold,
            scale=(1.2, 1.2, 0.1),
            position=(0, 0, 1.25),
            rotation=(90, 0, 0),
            shader=lit_with_shadows_shader
        )
        
        # Create fingers with joint segments
        self.fingers = []
        self.finger_joints = []
        
        # Create detailed fingers with joints
        for i in range(5):
            finger_segments = []
            
            # Three segments per finger
            for j in range(3):
                length = 0.8 - (j * 0.15)  # Decreasing length for each segment
                
                # Finger segment
                segment = Entity(
                    model=Mesh(vertices=[
                        Vec3(-0.1, 0, 0),          # Base left
                        Vec3(0.1, 0, 0),           # Base right
                        Vec3(0.08, 0, -length),    # Tip right
                        Vec3(-0.08, 0, -length),   # Tip left
                        Vec3(0, 0.1, 0),           # Base top
                        Vec3(0, 0.1, -length),     # Tip top
                        Vec3(0, -0.1, 0),          # Base bottom
                        Vec3(0, -0.1, -length),    # Tip bottom
                    ], triangles=[
                        0, 1, 4, 1, 4, 6,  # Base
                        2, 3, 5, 3, 5, 7,  # Tip
                        0, 3, 4, 3, 4, 5,  # Left
                        1, 2, 6, 2, 6, 7,  # Right
                        4, 5, 6, 5, 6, 7,  # Top
                        0, 1, 3, 1, 3, 2,  # Bottom
                    ]),
                    color=self.iron_man_red,
                    shader=lit_with_shadows_shader
                )
                
                # Gold joint connector
                joint = Entity(
                    model='sphere',
                    color=self.iron_man_gold,
                    scale=(0.12, 0.12, 0.12),
                    shader=lit_with_shadows_shader
                )
                
                finger_segments.append(segment)
                if j > 0:  # No joint at base of finger
                    self.finger_joints.append(joint)
            
            self.fingers.append(finger_segments)
        
        # Palm repulsor
        self.palm_repulsor_outer = Entity(
            model='circle',
            color=self.iron_man_gold,
            scale=(0.4, 0.4, 0.01),
            position=(0, 0, -0.8),
            shader=lit_with_shadows_shader
        )
        
        self.palm_repulsor_inner = Entity(
            model='circle',
            color=self.repulsor_blue,
            scale=(0.3, 0.3, 0.01),
            position=(0, 0, -0.81),
            shader=lit_with_shadows_shader
        )
        
        # Repulsor light effects
        self.palm_glow = Entity(
            model='sphere',
            color=color.rgba(30, 225, 255, 100),
            scale=(0.3, 0.3, 0.1),
            position=(0, 0, -0.85),
            shader=lit_with_shadows_shader
        )
        
        # Knuckle armor plates
        self.knuckle_plates = []
        for i in range(4):  # Skip thumb
            plate = Entity(
                model=Mesh(vertices=[
                    Vec3(-0.15, 0, 0),   # Base left
                    Vec3(0.15, 0, 0),    # Base right
                    Vec3(0.1, 0, -0.3),  # Tip right
                    Vec3(-0.1, 0, -0.3), # Tip left
                    Vec3(0, 0.1, 0),     # Base top
                    Vec3(0, 0.08, -0.3), # Tip top
                ], triangles=[
                    0, 1, 4,  # Base
                    2, 3, 5,  # Tip
                    0, 3, 4, 3, 4, 5,  # Left
                    1, 2, 4, 2, 4, 5,  # Right
                    0, 1, 3, 1, 3, 2,  # Bottom
                ]),
                color=self.iron_man_gold,
                shader=lit_with_shadows_shader
            )
            self.knuckle_plates.append(plate)
    
    def create_helmet_model(self):
        """Create detailed Iron Man helmet model using mesh generation"""
        # Main helmet shell
        self.helmet_base = Entity(
            model=Mesh(vertices=[
                # Bottom face vertices
                Vec3(-1.2, -1.0, 0),    # Bottom left back
                Vec3(1.2, -1.0, 0),     # Bottom right back
                Vec3(1.0, -0.8, -1.5),  # Bottom right front
                Vec3(-1.0, -0.8, -1.5), # Bottom left front
                
                # Middle face vertices
                Vec3(-1.3, 0, 0.3),     # Middle left back
                Vec3(1.3, 0, 0.3),      # Middle right back
                Vec3(1.1, 0, -1.7),     # Middle right front
                Vec3(-1.1, 0, -1.7),    # Middle left front
                
                # Top face vertices
                Vec3(-1.0, 1.2, 0),     # Top left back
                Vec3(1.0, 1.2, 0),      # Top right back
                Vec3(0.8, 1.0, -1.3),   # Top right front
                Vec3(-0.8, 1.0, -1.3),  # Top left front
            ], triangles=[
                # Bottom face
                0, 1, 2, 0, 2, 3,
                
                # Middle left face
                0, 3, 7, 0, 7, 4,
                
                # Middle right face
                1, 5, 6, 1, 6, 2,
                
                # Middle front face
                3, 2, 6, 3, 6, 7,
                
                # Middle back face
                0, 4, 5, 0, 5, 1,
                
                # Top left face
                4, 7, 11, 4, 11, 8,
                
                # Top right face
                5, 9, 10, 5, 10, 6,
                
                # Top front face
                7, 6, 10, 7, 10, 11,
                
                # Top back face
                4, 8, 9, 4, 9, 5,
                
                # Top face
                8, 11, 10, 8, 10, 9,
            ]),
            color=self.iron_man_red,
            shader=lit_with_shadows_shader
        )
        
        # Face plate
        self.face_plate = Entity(
            model=Mesh(vertices=[
                Vec3(-1.0, -0.7, -1.5),  # Bottom left
                Vec3(1.0, -0.7, -1.5),   # Bottom right
                Vec3(0.9, 0.7, -1.6),    # Top right
                Vec3(-0.9, 0.7, -1.6),   # Top left
                Vec3(-0.8, -0.5, -1.8),  # Inner bottom left
                Vec3(0.8, -0.5, -1.8),   # Inner bottom right
                Vec3(0.7, 0.5, -1.9),    # Inner top right
                Vec3(-0.7, 0.5, -1.9),   # Inner top left
            ], triangles=[
                0, 1, 5, 0, 5, 4,  # Bottom face
                1, 2, 6, 1, 6, 5,  # Right face
                2, 3, 7, 2, 7, 6,  # Top face
                3, 0, 4, 3, 4, 7,  # Left face
                4, 5, 6, 4, 6, 7,  # Inner face
            ]),
            color=self.iron_man_gold,
            shader=lit_with_shadows_shader
        )
        
        # Eye slits with glowing effect
        self.left_eye = Entity(
            model=Mesh(vertices=[
                Vec3(-0.8, 0.2, -1.81),  # Bottom left
                Vec3(-0.5, 0.2, -1.81),  # Bottom right
                Vec3(-0.45, 0.4, -1.81), # Top right
                Vec3(-0.75, 0.4, -1.81), # Top left
            ], triangles=[
                0, 1, 2, 0, 2, 3,  # Eye slit face
            ]),
            color=self.repulsor_blue,
            shader=lit_with_shadows_shader
        )
        
        self.right_eye = Entity(
            model=Mesh(vertices=[
                Vec3(0.5, 0.2, -1.81),   # Bottom left
                Vec3(0.8, 0.2, -1.81),   # Bottom right
                Vec3(0.75, 0.4, -1.81),  # Top right
                Vec3(0.45, 0.4, -1.81),  # Top left
            ], triangles=[
                0, 1, 2, 0, 2, 3,  # Eye slit face
            ]),
            color=self.repulsor_blue,
            shader=lit_with_shadows_shader
        )
        
        # Eye glow effects
        self.left_eye_glow = Entity(
            model='quad',
            color=color.rgba(30, 225, 255, 150),
            scale=(0.3, 0.2, 0.1),
            position=(-0.65, 0.3, -1.85),
            billboard=True,  # Always face camera
            shader=lit_with_shadows_shader
        )
        
        self.right_eye_glow = Entity(
            model='quad',
            color=color.rgba(30, 225, 255, 150),
            scale=(0.3, 0.2, 0.1),
            position=(0.65, 0.3, -1.85),
            billboard=True,  # Always face camera
            shader=lit_with_shadows_shader
        )
        
        # Helmet details - ridges and panels
        self.helmet_crest = Entity(
            model=Mesh(vertices=[
                Vec3(0, 1.2, -0.2),    # Top center
                Vec3(-0.2, 0.8, -0.3),  # Left
                Vec3(0.2, 0.8, -0.3),   # Right
                Vec3(0, 0.7, -1.0),     # Front
            ], triangles=[
                0, 1, 3,  # Left panel
                0, 3, 2,  # Right panel
                0, 2, 1,  # Back panel
            ]),
            color=self.iron_man_red,
            shader=lit_with_shadows_shader
        )
        
        # Ear pieces
        self.left_ear = Entity(
            model='sphere',
            color=self.iron_man_red,
            scale=(0.2, 0.3, 0.2),
            position=(-1.3, 0, -0.2),
            shader=lit_with_shadows_shader
        )
        
        self.right_ear = Entity(
            model='sphere',
            color=self.iron_man_red,
            scale=(0.2, 0.3, 0.2),
            position=(1.3, 0, -0.2),
            shader=lit_with_shadows_shader
        )
        
        # Jaw line details
        self.jaw_detail = Entity(
            model=Mesh(vertices=[
                Vec3(-1.0, -0.8, -0.2),  # Left back
                Vec3(1.0, -0.8, -0.2),   # Right back
                Vec3(0.9, -0.8, -1.3),   # Right front
                Vec3(-0.9, -0.8, -1.3),  # Left front
                Vec3(-1.0, -1.0, -0.2),  # Left back bottom
                Vec3(1.0, -1.0, -0.2),   # Right back bottom
                Vec3(0.9, -1.0, -1.3),   # Right front bottom
                Vec3(-0.9, -1.0, -1.3),  # Left front bottom
            ], triangles=[
                0, 1, 2, 0, 2, 3,  # Top
                4, 5, 6, 4, 6, 7,  # Bottom
                0, 3, 7, 0, 7, 4,  # Left
                1, 5, 6, 1, 6, 2,  # Right
                3, 2, 6, 3, 6, 7,  # Front
                0, 1, 5, 0, 5, 4,  # Back
            ]),
            color=self.iron_man_gold,
            shader=lit_with_shadows_shader
        )
        
        # Group all helmet components
        self.helmet_components = [
            self.helmet_base, self.face_plate, 
            self.left_eye, self.right_eye,
            self.left_eye_glow, self.right_eye_glow,
            self.helmet_crest, self.left_ear, self.right_ear,
            self.jaw_detail
        ]
    
    def create_chest_model(self):
        """Create Iron Man chest piece model"""
        # Main chest plate
        self.chest_plate = Entity(
            model=Mesh(vertices=[
                # Front face vertices
                Vec3(-2.0, -1.5, -1.0),  # Bottom left
                Vec3(2.0, -1.5, -1.0),   # Bottom right
                Vec3(2.5, 1.5, -1.0),    # Top right
                Vec3(-2.5, 1.5, -1.0),   # Top left
                
                # Back face vertices (slightly curved)
                Vec3(-1.8, -1.3, -2.0),  # Bottom left
                Vec3(1.8, -1.3, -2.0),   # Bottom right
                Vec3(2.3, 1.7, -2.0),    # Top right
                Vec3(-2.3, 1.7, -2.0),   # Top left
            ], triangles=[
                # Front face
                0, 1, 2, 0, 2, 3,
                
                # Back face
                4, 7, 6, 4, 6, 5,
                
                # Left face
                0, 3, 7, 0, 7, 4,
                
                # Right face
                1, 5, 6, 1, 6, 2,
                
                # Top face
                3, 2, 6, 3, 6, 7,
                
                # Bottom face
                0, 4, 5, 0, 5, 1,
            ]),
            color=self.iron_man_red,
            shader=lit_with_shadows_shader
        )
        
        # Arc reactor
        self.arc_reactor_outer = Entity(
            model='circle',
            color=self.iron_man_gold,
            scale=(0.8, 0.8, 0.1),
            position=(0, 0, -1.0),
            shader=lit_with_shadows_shader
        )
        
        self.arc_reactor_inner = Entity(
            model='circle',
            color=self.repulsor_blue,
            scale=(0.6, 0.6, 0.1),
            position=(0, 0, -0.95),
            shader=lit_with_shadows_shader
        )
        
        # Arc reactor glow
        self.arc_reactor_glow = Entity(
            model='sphere',
            color=color.rgba(30, 225, 255, 150),
            scale=(0.5, 0.5, 0.2),
            position=(0, 0, -0.9),
            shader=lit_with_shadows_shader
        )
        
        # Shoulder pieces
        self.left_shoulder = Entity(
            model='sphere',
            color=self.iron_man_red,
            scale=(0.8, 0.8, 0.8),
            position=(-2.5, 1.5, -1.5),
            shader=lit_with_shadows_shader
        )
        
        self.right_shoulder = Entity(
            model='sphere',
            color=self.iron_man_red,
            scale=(0.8, 0.8, 0.8),
            position=(2.5, 1.5, -1.5),
            shader=lit_with_shadows_shader
        )
        
        # Chest detail lines (gold trim)
        self.chest_details = Entity(
            model=Mesh(vertices=[
                # Left detail
                Vec3(-1.5, 0.5, -0.95),  # Top
                Vec3(-1.0, -1.0, -0.95), # Bottom
                Vec3(-1.3, -1.0, -0.95), # Bottom wider
                Vec3(-1.8, 0.5, -0.95),  # Top wider
                
                # Right detail (mirrored)
                Vec3(1.5, 0.5, -0.95),   # Top
                Vec3(1.0, -1.0, -0.95),  # Bottom
                Vec3(1.3, -1.0, -0.95),  # Bottom wider
                Vec3(1.8, 0.5, -0.95),   # Top wider
            ], triangles=[
                # Left detail
                0, 1, 2, 0, 2, 3,
                
                # Right detail
                4, 5, 6, 4, 6, 7,
            ]),
            color=self.iron_man_gold,
            shader=lit_with_shadows_shader
        )
        
        # Group all chest components
        self.chest_components = [
            self.chest_plate, self.arc_reactor_outer, 
            self.arc_reactor_inner, self.arc_reactor_glow,
            self.left_shoulder, self.right_shoulder,
            self.chest_details
        ]
        
        # Initially hide chest components
        for component in self.chest_components:
            component.visible = False
    
    def create_hud_elements(self):
        """Create HUD interface elements"""
        # Create targeting reticle
        self.targeting_reticle = Entity(
            model=Mesh(vertices=[
                # Inner circle vertices
                Vec3(0, 0, -5),             # Center
                Vec3(0.2, 0, -5),           # Right
                Vec3(0.14, 0.14, -5),       # Top right
                Vec3(0, 0.2, -5),           # Top
                Vec3(-0.14, 0.14, -5),      # Top left
                Vec3(-0.2, 0, -5),          # Left
                Vec3(-0.14, -0.14, -5),     # Bottom left
                Vec3(0, -0.2, -5),          # Bottom
                Vec3(0.14, -0.14, -5),      # Bottom right
                
                # Outer circle vertices
                Vec3(0.3, 0, -5),           # Right
                Vec3(0.21, 0.21, -5),       # Top right
                Vec3(0, 0.3, -5),           # Top
                Vec3(-0.21, 0.21, -5),      # Top left
                Vec3(-0.3, 0, -5),          # Left
                Vec3(-0.21, -0.21, -5),     # Bottom left
                Vec3(0, -0.3, -5),          # Bottom
                Vec3(0.21, -0.21, -5),      # Bottom right
            ], triangles=[
                # Inner circle segments
                0, 1, 2,
                0, 2, 3,
                0, 3, 4, 
                0, 4, 5,
                0, 5, 6,
                0, 6, 7,
                0, 7, 8,
                0, 8, 1,
                
                # Outer ring segments
                1, 9, 10, 1, 10, 2,
                2, 10, 11, 2, 11, 3,
                3, 11, 12, 3, 12, 4,
                4, 12, 13, 4, 13, 5,
                5, 13, 14, 5, 14, 6,
                6, 14, 15, 6, 15, 7,
                7, 15, 16, 7, 16, 8,
                8, 16, 9, 8, 9, 1,
            ]),
            color=self.repulsor_blue,
            billboard=True,
            shader=lit_with_shadows_shader,
            visible=False  # Initially hidden
        )
        
        # Add crosshair lines
        self.crosshair_h = Entity(
            model=Mesh(vertices=[
                Vec3(-0.5, 0, -5),
                Vec3(0.5, 0, -5),
                Vec3(-0.5, 0.02, -5),
                Vec3(0.5, 0.02, -5),
            ], triangles=[
                0, 1, 3, 0, 3, 2,
            ]),
            color=self.repulsor_blue,
            billboard=True,
            shader=lit_with_shadows_shader,
            visible=False
        )
        
        self.crosshair_v = Entity(
            model=Mesh(vertices=[
                Vec3(0, -0.5, -5),
                Vec3(0, 0.5, -5),
                Vec3(0.02, -0.5, -5),
                Vec3(0.02, 0.5, -5),
            ], triangles=[
                0, 1, 3, 0, 3, 2,
            ]),
            color=self.repulsor_blue,
            billboard=True,
            shader=lit_with_shadows_shader,
            visible=False
        )
        
        # Data display panel
        self.data_panel = Entity(
            model='quad',
            texture='white_cube',
            color=color.rgba(0, 0, 0, 100),
            scale=(3, 1.5, 1),
            position=(5, 3, -5),
            visible=False
        )
        
        # Group HUD elements
        self.hud_elements = [
            self.targeting_reticle,
            self.crosshair_h,
            self.crosshair_v,
            self.data_panel
        ]
    
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
        
        # Update repulsor particles
        self.update_repulsor_particles()
        
        # Update hand model position and animation
        self.update_hand_model()
        
        # Update helmet position and animation
        self.update_helmet_model()
        
        # Update chest piece position and animation
        self.update_chest_model()
        
        # Update HUD elements
        self.update_hud()
    
    def update_hand_model(self):
        """Update hand model position and animation based on tracking"""
        if not self.hand_landmarks:
            # Hide hand components when hand is not visible
            self.gauntlet_base.visible = False
            self.wrist_armor.visible = False
            self.wrist_trim.visible = False
            self.palm_repulsor_outer.visible = False
            self.palm_repulsor_inner.visible = False
            self.palm_glow.visible = False
            
            for segments in self.fingers:
                for segment in segments:
                    segment.visible = False
            
            for joint in self.finger_joints:
                joint.visible = False
                
            for plate in self.knuckle_plates:
                plate.visible = False
                
            return
        
        # Position the gauntlet based on hand position
        hand_x = (self.hand_landmarks.landmark[0].x - 0.5) * 20
        hand_y = (0.5 - self.hand_landmarks.landmark[0].y) * 20
        hand_z = -5 - (self.hand_landmarks.landmark[0].z * 10)
        
        # Apply hand dimensions to scale
        width_scale = max(0.8, min(1.5, self.hand_dimensions['width'] / 200))
        length_scale = max(0.8, min(1.5, self.hand_dimensions['length'] / 200))
        
        # Calculate hand rotation based on wrist and middle finger
        wrist = self.hand_landmarks.landmark[0]
        middle_base = self.hand_landmarks.landmark[9]
        
        # Calculate hand rotation
        dx = middle_base.x - wrist.x
        dy = middle_base.y - wrist.y
        
        hand_angle_y = math.degrees(math.atan2(dx, -dy))
        hand_angle_x = math.degrees(math.atan2(wrist.z - middle_base.z, 
                                            math.sqrt((middle_base.x - wrist.x)**2 + (middle_base.y - wrist.y)**2)))
        
        # Update main gauntlet components
        alpha = int(255 * self.deployment_progress)
        visibility = self.deployment_progress > 0.1
        
        self.gauntlet_base.visible = visibility
        self.gauntlet_base.position = Vec3(hand_x, hand_y, hand_z)
        self.gauntlet_base.rotation = Vec3(hand_angle_x, 0, hand_angle_y)
        self.gauntlet_base.scale = Vec3(2 * width_scale, 0.5, 1.5 * length_scale)
        self.gauntlet_base.color = color.rgba(self.iron_man_red.r, self.iron_man_red.g, self.iron_man_red.b, alpha)
        
        self.wrist_armor.visible = visibility
        self.wrist_armor.position = Vec3(hand_x, hand_y, hand_z)
        self.wrist_armor.rotation = Vec3(hand_angle_x, 0, hand_angle_y)
        self.wrist_armor.scale = Vec3(2 * width_scale, 0.5, 1.5 * length_scale)
        self.wrist_armor.color = color.rgba(self.iron_man_red.r, self.iron_man_red.g, self.iron_man_red.b, alpha)
        
        self.wrist_trim.visible = visibility
        self.wrist_trim.position = Vec3(hand_x, hand_y, hand_z + 0.25)
        self.wrist_trim.rotation = Vec3(90 + hand_angle_x, 0, hand_angle_y)
        self.wrist_trim.scale = Vec3(1.2 * width_scale, 1.2 * width_scale, 0.1)
        self.wrist_trim.color = color.rgba(self.iron_man_gold.r, self.iron_man_gold.g, self.iron_man_gold.b, alpha)
        
        # Update palm repulsors
        self.palm_repulsor_outer.visible = visibility
        self.palm_repulsor_outer.position = Vec3(hand_x, hand_y, hand_z - 0.8)
        self.palm_repulsor_outer.rotation = Vec3(hand_angle_x, 0, hand_angle_y)
        self.palm_repulsor_outer.scale = Vec3(0.4 * width_scale, 0.4 * width_scale, 0.01)
        self.palm_repulsor_outer.color = color.rgba(self.iron_man_gold.r, self.iron_man_gold.g, self.iron_man_gold.b, alpha)
        
        self.palm_repulsor_inner.visible = visibility
        self.palm_repulsor_inner.position = Vec3(hand_x, hand_y, hand_z - 0.81)
        self.palm_repulsor_inner.rotation = Vec3(hand_angle_x, 0, hand_angle_y)
        self.palm_repulsor_inner.scale = Vec3(0.3 * width_scale, 0.3 * width_scale, 0.01)
        
        # Make repulsor glow based on charge level
        glow_intensity = 100 + int(155 * (self.repulsor_charge_level / 100))
        self.palm_repulsor_inner.color = color.rgba(self.repulsor_blue.r, self.repulsor_blue.g, self.repulsor_blue.b, min(255, alpha + glow_intensity))
        
        # Update repulsor glow effect
        glow_pulse = math.sin(time.time() * 10) * 0.1
        glow_scale = 0.3 + glow_pulse + 0.3 * (self.repulsor_charge_level / 100)
        
        self.palm_glow.visible = visibility and self.repulsor_charge_level > 0
        self.palm_glow.position = Vec3(hand_x, hand_y, hand_z - 0.85)
        self.palm_glow.scale = Vec3(glow_scale * width_scale, glow_scale * width_scale, 0.1)
        self.palm_glow.color = color.rgba(30, 225, 255, min(200, 50 + int(150 * (self.repulsor_charge_level / 100))))
        
        # Update fingers
        for i, finger_segments in enumerate(self.fingers):
            # Get finger base, middle, and tip landmarks
            if i == 0:  # Thumb
                base_id, middle_id, tip_id = 1, 2, 4
            else:
                base_id = 5 + (i-1)*4
                middle_id = 6 + (i-1)*4
                tip_id = 8 + (i-1)*4
            
            base = self.hand_landmarks.landmark[base_id]
            middle = self.hand_landmarks.landmark[middle_id]
            tip = self.hand_landmarks.landmark[tip_id]
            
            # Calculate segment positions in 3D space
            base_pos = Vec3(
                (base.x - 0.5) * 20, 
                (0.5 - base.y) * 20, 
                -5 - (base.z * 10)
            )
            
            middle_pos = Vec3(
                (middle.x - 0.5) * 20, 
                (0.5 - middle.y) * 20, 
                -5 - (middle.z * 10)
            )
            
            tip_pos = Vec3(
                (tip.x - 0.5) * 20, 
                (0.5 - tip.y) * 20, 
                -5 - (tip.z * 10)
            )
            
            # Position segments along the finger
            segment_positions = [
                base_pos,
                Vec3.lerp(base_pos, middle_pos, 0.67),
                Vec3.lerp(middle_pos, tip_pos, 0.5)
            ]
            
            # Direction vectors for each segment
            directions = [
                (middle_pos - base_pos).normalized(),
                (middle_pos - base_pos).normalized(),
                (tip_pos - middle_pos).normalized()
            ]
            
            # Update each segment
            for j, segment in enumerate(finger_segments):
                pos = segment_positions[j]
                dir_vec = directions[j]
                
                # Calculate segment rotation
                up_vec = Vec3(0, 1, 0)
                right_vec = up_vec.cross(dir_vec).normalized()
                up_rot = right_vec.cross(dir_vec).normalized()
                
                # Convert direction to rotation angles
                rot_matrix = Matrix44(
                    right_vec.x, right_vec.y, right_vec.z, 0,
                    up_rot.x, up_rot.y, up_rot.z, 0,
                    dir_vec.x, dir_vec.y, dir_vec.z, 0,
                    0, 0, 0, 1
                )
                
                rotation = rot_matrix.get_euler_angles()
                
                # Update segment
                segment.visible = visibility
                segment.position = pos
                segment.rotation = Vec3(rotation.x, rotation.y, rotation.z)
                segment.color = color.rgba(self.iron_man_red.r, self.iron_man_red.g, self.iron_man_red.b, alpha)
            
            # Update finger joints
            if i > 0:  # Skip thumb joints
                knuckle_pos = Vec3.lerp(base_pos, middle_pos, 0.1)
                middle_joint_pos = Vec3.lerp(base_pos, middle_pos, 0.67)
                
                # Update knuckle plate
                self.knuckle_plates[i-1].visible = visibility
                self.knuckle_plates[i-1].position = knuckle_pos
                self.knuckle_plates[i-1].rotation = Vec3(rotation.x, rotation.y, rotation.z)
                self.knuckle_plates[i-1].color = color.rgba(self.iron_man_gold.r, self.iron_man_gold.g, self.iron_man_gold.b, alpha)
                
                # Update middle joint
                joint_index = (i-1) * 2
                if joint_index < len(self.finger_joints):
                    self.finger_joints[joint_index].visible = visibility
                    self.finger_joints[joint_index].position = middle_joint_pos
                    self.finger_joints[joint_index].color = color.rgba(self.iron_man_gold.r, self.iron_man_gold.g, self.iron_man_gold.b, alpha)
                
                # Update tip joint
                tip_joint_pos = Vec3.lerp(middle_pos, tip_pos, 0.5)
                if joint_index + 1 < len(self.finger_joints):
                    self.finger_joints[joint_index + 1].visible = visibility
                    self.finger_joints[joint_index + 1].position = tip_joint_pos
                    self.finger_joints[joint_index + 1].color = color.rgba(self.iron_man_gold.r, self.iron_man_gold.g, self.iron_man_gold.b, alpha)
    
    def update_helmet_model(self):
        """Update helmet model position and animation based on face tracking"""
        if not self.face_landmarks:
            # Hide helmet when face is not visible
            for component in self.helmet_components:
                component.visible = False
            return
        
        # Calculate center of the face
        nose_tip = self.face_landmarks.landmark[4]
        face_x = (nose_tip.x - 0.5) * 20
        face_y = (0.5 - nose_tip.y) * 20 + 2  # Offset to place helmet correctly
        face_z = -5 - (nose_tip.z * 10)
        
        # Scale based on face dimensions
        width_scale = max(0.8, min(1.5, self.face_dimensions['width'] / 200))
        height_scale = max(0.8, min(1.5, self.face_dimensions['height'] / 200))
        
        # Calculate face rotation using landmarks
        left_eye = self.face_landmarks.landmark[33]
        right_eye = self.face_landmarks.landmark[263]
        
        # Calculate face tilt (roll)
        dx = right_eye.x - left_eye.x
        dy = right_eye.y - left_eye.y
        face_tilt = math.degrees(math.atan2(dy, dx))
        
        # Calculate face yaw using nose and ear landmarks
        nose_bridge = self.face_landmarks.landmark[168]
        left_ear = self.face_landmarks.landmark[234]
        right_ear = self.face_landmarks.landmark[454]
        
        left_dist = math.sqrt((nose_bridge.x - left_ear.x)**2 + (nose_bridge.z - left_ear.z)**2)
        right_dist = math.sqrt((nose_bridge.x - right_ear.x)**2 + (nose_bridge.z - right_ear.z)**2)
        
        # Estimate yaw based on differential distance to ears
        face_yaw = (right_dist - left_dist) * 45
        
        # Calculate face pitch using vertical landmarks
        forehead = self.face_landmarks.landmark[10]
        chin = self.face_landmarks.landmark[152]
        
        # Use z-difference for pitch
        face_pitch = (nose_tip.z - forehead.z) * 100
        
        # Apply deployment animation
        alpha = int(255 * self.helmet_deployment_progress)
        visibility = self.helmet_deployment_progress > 0.1
        
        # Update all helmet components
        for component in self.helmet_components:
            component.visible = visibility
            component.position = Vec3(face_x, face_y, face_z)
            component.rotation = Vec3(face_pitch, face_yaw, face_tilt)
            
            # Apply alpha for deployment animation
            if hasattr(component, 'color') and hasattr(component.color, 'a'):
                r, g, b = component.color.r, component.color.g, component.color.b
                component.color = color.rgba(r, g, b, alpha)
        
        # Scale helmet components
        self.helmet_base.scale = Vec3(3 * width_scale, 3 * height_scale, 3 * width_scale)
        self.face_plate.scale = Vec3(2.8 * width_scale, 2.8 * height_scale, 0.5)
        self.helmet_crest.scale = Vec3(0.5 * width_scale, 0.8 * height_scale, 0.5)
        self.left_ear.scale = Vec3(0.2 * width_scale, 0.3 * height_scale, 0.2)
        self.right_ear.scale = Vec3(0.2 * width_scale, 0.3 * height_scale, 0.2)
        
        # Update eye glow effect
        glow_pulse = math.sin(time.time() * 5) * 0.1
        eye_glow_scale = 0.25 + glow_pulse
        
        self.left_eye_glow.scale = Vec3(eye_glow_scale * width_scale, eye_glow_scale * height_scale, 0.1)
        self.left_eye_glow.color = color.rgba(30, 225, 255, min(200, 50 + int(150 * self.helmet_deployment_progress)))
        
        self.right_eye_glow.scale = Vec3(eye_glow_scale * width_scale, eye_glow_scale * height_scale, 0.1)
        self.right_eye_glow.color = color.rgba(30, 225, 255, min(200, 50 + int(150 * self.helmet_deployment_progress)))

    def update_chest_model(self):
        """Update chest model position and animation based on pose tracking"""
        if not self.pose_landmarks:
            # Hide chest components when pose is not visible
            for component in self.chest_components:
                component.visible = False
            return
        
        # Calculate chest position using shoulder landmarks
        left_shoulder = self.pose_landmarks.landmark[11]
        right_shoulder = self.pose_landmarks.landmark[12]
        chest_center = self.pose_landmarks.landmark[0]  # Nose as reference for depth
        
        # Calculate position
        chest_x = ((left_shoulder.x + right_shoulder.x) / 2 - 0.5) * 20
        chest_y = (0.5 - (left_shoulder.y + right_shoulder.y) / 2) * 20 - 2  # Offset to place correctly
        chest_z = -5 - (chest_center.z * 10) - 2  # Offset to place behind the head
        
        # Calculate chest dimensions
        shoulder_width = abs(right_shoulder.x - left_shoulder.x) * 20
        width_scale = max(0.8, min(1.5, shoulder_width / 6))
        
        # Calculate chest rotation (if both shoulders visible)
        chest_roll = 0
        if left_shoulder and right_shoulder:
            dx = right_shoulder.x - left_shoulder.x
            dy = right_shoulder.y - left_shoulder.y
            chest_roll = math.degrees(math.atan2(dy, dx))
        
        # Apply deployment animation
        alpha = int(255 * self.chest_deployment_progress)
        visibility = self.chest_deployment_progress > 0.1
        
        # Update all chest components
        for component in self.chest_components:
            component.visible = visibility
            component.position = Vec3(chest_x, chest_y, chest_z)
            component.rotation = Vec3(0, 0, chest_roll)
            
            # Apply alpha for deployment animation
            if hasattr(component, 'color') and hasattr(component.color, 'a'):
                r, g, b = component.color.r, component.color.g, component.color.b
                component.color = color.rgba(r, g, b, alpha)
        
        # Scale chest components
        self.chest_plate.scale = Vec3(width_scale, width_scale, width_scale)
        self.left_shoulder.scale = Vec3(0.8 * width_scale, 0.8 * width_scale, 0.8 * width_scale)
        self.right_shoulder.scale = Vec3(0.8 * width_scale, 0.8 * width_scale, 0.8 * width_scale)
        
        # Update arc reactor glow
        glow_pulse = math.sin(time.time() * 3) * 0.1
        arc_glow_scale = 0.5 + glow_pulse
        
        self.arc_reactor_glow.scale = Vec3(arc_glow_scale * width_scale, arc_glow_scale * width_scale, 0.2)
        self.arc_reactor_glow.color = color.rgba(30, 225, 255, min(200, 50 + int(150 * self.chest_deployment_progress)))

    def update_repulsor_particles(self):
        """Update repulsor particle effects"""
        # Only show particles when repulsor is charging
        if self.repulsor_charge_level > 10 and self.palm_glow.visible:
            # Create new particles at the palm position
            for _ in range(int(self.repulsor_charge_level / 20)):
                if random.random() < 0.5:  # Randomly create particles
                    # Get palm position
                    palm_pos = Vec3(
                        self.palm_glow.x,
                        self.palm_glow.y,
                        self.palm_glow.z - 0.1
                    )
                    
                    # Create a particle
                    particle = Entity(
                        model='sphere',
                        color=color.rgba(30, 225, 255, 150),
                        scale=0.05,
                        position=palm_pos,
                        billboard=True
                    )
                    
                    # Add random velocity
                    velocity = Vec3(
                        random.uniform(-0.1, 0.1),
                        random.uniform(-0.1, 0.1),
                        random.uniform(-0.2, -0.4)
                    )
                    
                    # Add to particles list with lifespan and velocity
                    self.particles.append({
                        'entity': particle,
                        'velocity': velocity,
                        'life': 1.0  # 1 second lifespan
                    })
        
        # Update existing particles
        particles_to_remove = []
        for particle in self.particles:
            # Update position
            particle['entity'].position += particle['velocity'] * time.dt
            
            # Update life
            particle['life'] -= time.dt
            
            # Update scale and alpha based on life
            life_ratio = particle['life']
            particle['entity'].scale = Vec3(0.05 * life_ratio, 0.05 * life_ratio, 0.05 * life_ratio)
            particle['entity'].color = color.rgba(30, 225, 255, int(150 * life_ratio))
            
            # Mark for removal if expired
            if particle['life'] <= 0:
                particles_to_remove.append(particle)
        
        # Remove expired particles
        for particle in particles_to_remove:
            if particle['entity'] in scene.entities:
                destroy(particle['entity'])
            self.particles.remove(particle)

    def update_hud(self):
        """Update HUD elements based on tracking and system status"""
        # Show targeting reticle if hand is visible and in repulsor pose
        if self.hand_landmarks and self.repulsor_charge_level > 0:
            self.targeting_reticle.visible = True
            self.crosshair_h.visible = True
            self.crosshair_v.visible = True
            
            # Position reticle in front of palm
            if self.palm_glow.visible:
                reticle_x = self.palm_glow.x + (math.sin(time.time() * 5) * 0.05)
                reticle_y = self.palm_glow.y + (math.cos(time.time() * 5) * 0.05)
                reticle_z = self.palm_glow.z - 3
                
                self.targeting_reticle.position = Vec3(reticle_x, reticle_y, reticle_z)
                self.crosshair_h.position = Vec3(reticle_x, reticle_y, reticle_z)
                self.crosshair_v.position = Vec3(reticle_x, reticle_y, reticle_z)
                
                # Scale based on distance
                scale_factor = 0.5 + (self.repulsor_charge_level / 200)
                self.targeting_reticle.scale = Vec3(scale_factor, scale_factor, 1)
                self.crosshair_h.scale = Vec3(scale_factor * 2, 0.05, 1)
                self.crosshair_v.scale = Vec3(0.05, scale_factor * 2, 1)
        else:
            self.targeting_reticle.visible = False
            self.crosshair_h.visible = False
            self.crosshair_v.visible = False
        
        # Show data panel if all tracking is active
        if self.hand_landmarks and self.face_landmarks and self.pose_landmarks:
            self.data_panel.visible = True
            
            # Update data panel information
            # (In a full implementation, this would update Text entities with system status)
        else:
            self.data_panel.visible = False

    def process_gestures(self):
        """Process hand gestures to control the system"""
        if not self.hand_landmarks:
            return
        
        # Detect repulsor gesture (open palm with fingers extended)
        is_palm_open = self.detect_open_palm()
        
        # Update repulsor charge level based on gesture
        if is_palm_open:
            self.repulsor_charge_level = min(100, self.repulsor_charge_level + 2)
        else:
            self.repulsor_charge_level = max(0, self.repulsor_charge_level - 5)
        
        # Detect helmet deployment gesture (closed fist near face)
        if self.detect_fist() and self.face_landmarks:
            # Calculate distance between fist and face
            hand_pos = self.hand_landmarks.landmark[9]  # Middle finger base
            face_pos = self.face_landmarks.landmark[4]  # Nose tip
            
            distance = math.sqrt(
                (hand_pos.x - face_pos.x)**2 + 
                (hand_pos.y - face_pos.y)**2 + 
                (hand_pos.z - face_pos.z)**2
            )
            
            if distance < 0.15:  # Close to face
                self.helmet_deployment_progress = min(1.0, self.helmet_deployment_progress + 0.05)
            else:
                self.helmet_deployment_progress = max(0.0, self.helmet_deployment_progress - 0.02)
        else:
            self.helmet_deployment_progress = max(0.0, self.helmet_deployment_progress - 0.01)
        
        # Detect chest deployment gesture (both hands on chest)
        if self.pose_landmarks and self.hand_landmarks:
            # Check if hand is near chest
            hand_pos = self.hand_landmarks.landmark[0]  # Wrist
            chest_pos = self.pose_landmarks.landmark[0]  # Nose (approximation)
            
            distance = math.sqrt(
                (hand_pos.x - chest_pos.x)**2 + 
                (hand_pos.y - chest_pos.y)**2 + 
                (hand_pos.z - chest_pos.z)**2
            )
            
            if distance < 0.3:  # Close to chest
                self.chest_deployment_progress = min(1.0, self.chest_deployment_progress + 0.05)
            else:
                self.chest_deployment_progress = max(0.0, self.chest_deployment_progress - 0.02)
        else:
            self.chest_deployment_progress = max(0.0, self.chest_deployment_progress - 0.01)

    def detect_open_palm(self):
        """Detect if hand is in open palm position for repulsor"""
        if not self.hand_landmarks:
            return False
        
        # Check finger extension
        landmarks = self.hand_landmarks.landmark
        
        # Check if fingers are extended
        fingerTips = [4, 8, 12, 16, 20]
        fingerIPs = [3, 7, 11, 15, 19]  # Intermediate points
        fingerMPs = [2, 6, 10, 14, 18]  # Base points
        
        # Check if fingers are extended
        fingers_extended = 0
        for tip, ip, mp in zip(fingerTips, fingerIPs, fingerMPs):
            # Calculate vectors
            vec1 = Vec3(
                landmarks[ip].x - landmarks[mp].x,
                landmarks[ip].y - landmarks[mp].y,
                landmarks[ip].z - landmarks[mp].z
            )
            
            vec2 = Vec3(
                landmarks[tip].x - landmarks[ip].x,
                landmarks[tip].y - landmarks[ip].y,
                landmarks[tip].z - landmarks[ip].z
            )
            
            # Check if finger is extended (vectors aligned)
            dot_product = vec1.dot(vec2)
            vec1_len = vec1.length()
            vec2_len = vec2.length()
            
            if vec1_len > 0 and vec2_len > 0:
                alignment = dot_product / (vec1_len * vec2_len)
                if alignment > 0.7:  # Aligned = extended
                    fingers_extended += 1
        
        # Return true if at least 4 fingers are extended
        return fingers_extended >= 4

    def detect_fist(self):
        """Detect closed fist gesture"""
        if not self.hand_landmarks:
            return False
        
        # Check finger flexion (opposite of extension)
        landmarks = self.hand_landmarks.landmark
        
        # Check if fingers are curled
        fingerTips = [4, 8, 12, 16, 20]
        fingerPIPs = [3, 7, 11, 15, 19]  # Second joint
        fingerMCPs = [2, 6, 10, 14, 18]  # Base points
        
        # Count curled fingers
        fingers_curled = 0
        for tip, pip, mcp in zip(fingerTips, fingerPIPs, fingerMCPs):
            # Calculate vectors
            vec1 = Vec3(
                landmarks[pip].x - landmarks[mcp].x,
                landmarks[pip].y - landmarks[mcp].y,
                landmarks[pip].z - landmarks[mcp].z
            )
            
            vec2 = Vec3(
                landmarks[tip].x - landmarks[pip].x,
                landmarks[tip].y - landmarks[pip].y,
                landmarks[tip].z - landmarks[pip].z
            )
            
            # Check if finger is curled (vectors perpendicular or opposite)
            dot_product = vec1.dot(vec2)
            vec1_len = vec1.length()
            vec2_len = vec2.length()
            
            if vec1_len > 0 and vec2_len > 0:
                alignment = dot_product / (vec1_len * vec2_len)
                if alignment < 0.3:  # Perpendicular or opposite = curled
                    fingers_curled += 1
        
        # Return true if at least 4 fingers are curled
        return fingers_curled >= 4

    def camera_callback(self, frame):
        """Callback function for camera frame"""
        if frame is None:
            return
        
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hands
        hand_results = self.hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            self.hand_landmarks = hand_results.multi_hand_landmarks[0]  # Use first hand
            
            # Calculate hand dimensions
            landmarks = self.hand_landmarks.landmark
            min_x, max_x = float('inf'), float('-inf')
            min_y, max_y = float('inf'), float('-inf')
            
            for lm in landmarks:
                min_x = min(min_x, lm.x)
                max_x = max(max_x, lm.x)
                min_y = min(min_y, lm.y)
                max_y = max(max_y, lm.y)
            
            width = (max_x - min_x) * frame.shape[1]
            length = (max_y - min_y) * frame.shape[0]
            
            self.hand_dimensions = {
                'width': width,
                'length': length
            }
        else:
            self.hand_landmarks = None
        
        # Process face mesh
        face_results = self.face_mesh.process(rgb_frame)
        if face_results.multi_face_landmarks:
            self.face_landmarks = face_results.multi_face_landmarks[0]  # Use first face
            
            # Calculate face dimensions
            landmarks = self.face_landmarks.landmark
            min_x, max_x = float('inf'), float('-inf')
            min_y, max_y = float('inf'), float('-inf')
            
            for lm in landmarks:
                min_x = min(min_x, lm.x)
                max_x = max(max_x, lm.x)
                min_y = min(min_y, lm.y)
                max_y = max(max_y, lm.y)
            
            width = (max_x - min_x) * frame.shape[1]
            height = (max_y - min_y) * frame.shape[0]
            
            self.face_dimensions = {
                'width': width,
                'height': height
            }
        else:
            self.face_landmarks = None
        
        # Process pose
        pose_results = self.pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            self.pose_landmarks = pose_results.pose_landmarks
        else:
            self.pose_landmarks = None
        
        # Process gestures
        self.process_gestures()
        
        # Store frame for display
        self.frame = frame
        self.frame_ready = True

    def start_camera(self):
        """Initialize and start camera capture"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=self.camera_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()

    def camera_loop(self):
        """Main camera capture loop"""
        while True:
            ret, frame = self.cap.read()
            if ret:
                self.camera_callback(frame)
            time.sleep(0.01)  # Small delay to reduce CPU usage

    def cleanup(self):
        """Clean up resources on exit"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()


# Main program entry point
if __name__ == "__main__":
    # Initialize Ursina app
    app = Ursina()
    
    # Create Iron Man AR interface
    iron_man_ar = IronManARInterface()
    
    # Start camera capture
    iron_man_ar.start_camera()
    
    # Register cleanup handler
    import atexit
    atexit.register(iron_man_ar.cleanup)
    
    # Run the app
    app.run()
