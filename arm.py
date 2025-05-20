import cv2
import mediapipe as mp
import numpy as np
import math
from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
import time
import random

# Initialize Ursina application
app = Ursina(borderless=False)
window.size = (1280, 720)
window.title = "Advanced Robotic Arm Simulation"

# Set up camera and lighting
camera.position = (0, 2, -8)
camera.rotation = (15, 0, 0)
DirectionalLight(parent=scene, y=2, z=3, shadows=True, rotation=(45, -45, 45))
AmbientLight(parent=scene, color=color.rgba(100, 100, 100, 0.1))

# Create environment
environment = Entity(
    model='sphere',
    texture='textures/skybox',
    scale=500,
    double_sided=True
)

# Create a platform for the robotic arm
platform = Entity(
    model='cube',
    color=color.dark_gray,
    scale=(3, 0.2, 3),
    position=(0, 0, 0),
    texture='white_cube',
    texture_scale=(3, 3),
    collider='box'
)

# Create robotic arm components
class RoboticArm:
    def __init__(self):
        # Base
        self.base = Entity(
            model='cylinder',
            color=color.dark_gray,
            scale=(1, 0.2, 1),
            position=(0, 0.2, 0),
            texture='white_cube'
        )
        
        # Rotating joint
        self.rotating_joint = Entity(
            model='sphere',
            color=color.gray,
            scale=0.5,
            position=(0, 0.5, 0),
            parent=self.base
        )
        
        # Lower arm
        self.lower_arm = Entity(
            model='cube',
            color=color.light_gray,
            scale=(0.3, 1.5, 0.3),
            position=(0, 0.8, 0),
            parent=self.rotating_joint
        )
        
        # Mid joint
        self.mid_joint = Entity(
            model='sphere',
            color=color.gray,
            scale=0.4,
            position=(0, 1.5, 0),
            parent=self.lower_arm
        )
        
        # Upper arm
        self.upper_arm = Entity(
            model='cube',
            color=color.light_gray,
            scale=(0.25, 1.3, 0.25),
            position=(0, 0.7, 0),
            parent=self.mid_joint
        )
        
        # Wrist joint
        self.wrist_joint = Entity(
            model='sphere',
            color=color.gray,
            scale=0.3,
            position=(0, 1.3, 0),
            parent=self.upper_arm
        )
        
        # Hand
        self.hand = Entity(
            model='cube',
            color=color.white,
            scale=(0.4, 0.2, 0.6),
            position=(0, 0.3, 0),
            parent=self.wrist_joint
        )
        
        # Fingers
        self.fingers = []
        for i in range(3):
            finger = Entity(
                model='cube',
                color=color.white,
                scale=(0.08, 0.4, 0.08),
                position=(-0.15 + i*0.15, 0.2, 0.25),
                parent=self.hand
            )
            self.fingers.append(finger)
        
        # Thumb
        self.thumb = Entity(
            model='cube',
            color=color.white,
            scale=(0.08, 0.3, 0.08),
            position=(-0.25, 0.1, 0),
            rotation=(0, 0, 45),
            parent=self.hand
        )
        
        # Add details to the arm
        self.details = []
        # Hydraulic cylinders
        hydraulic1 = Entity(
            model='cylinder',
            color=color.dark_gray,
            scale=(0.05, 1.2, 0.05),
            position=(0.15, 0.6, 0.15),
            rotation=(0, 0, 15),
            parent=self.lower_arm
        )
        self.details.append(hydraulic1)
        
        hydraulic2 = Entity(
            model='cylinder',
            color=color.dark_gray,
            scale=(0.05, 1, 0.05),
            position=(0.15, 0.5, 0.15),
            rotation=(0, 0, 15),
            parent=self.upper_arm
        )
        self.details.append(hydraulic2)
        
        # Wires
        for i in range(5):
            wire = Entity(
                model='cylinder',
                color=color.random_color(),
                scale=(0.02, 3, 0.02),
                position=(-0.2 + i*0.02, 0.5, -0.2),
                parent=self.lower_arm
            )
            self.details.append(wire)
        
        # Palm repulsor
        self.repulsor = Entity(
            model='sphere',
            color=color.rgba(0, 0.8, 0.8, 0.7),
            scale=0.15,
            position=(0, 0, 0.31),
            parent=self.hand
        )
        
        # Repulsor light
        self.repulsor_light = PointLight(
            parent=self.repulsor,
            color=color.cyan,
            intensity=0.5
        )
        self.repulsor_light.visible = False
        
        # Repulsor beam
        self.repulsor_beam = Entity(
            model='cylinder',
            color=color.rgba(0, 0.8, 0.8, 0.5),
            scale=(0.1, 15, 0.1),
            position=(0, 7.5, 0),
            rotation=(90, 0, 0),
            parent=self.repulsor
        )
        self.repulsor_beam.visible = False
        
        # Particles for the repulsor
        self.particles = []
        
    def update_position(self, wrist_pos, elbow_pos, shoulder_pos, hand_angle):
        # Scale and adjust positions to fit the 3D space
        scale_factor = 5
        
        # Update base rotation to track hand movement on horizontal plane
        target_x = wrist_pos[0] * scale_factor
        target_z = wrist_pos[2] * scale_factor
        
        base_rotation = math.degrees(math.atan2(target_x, target_z))
        self.rotating_joint.rotation_y = base_rotation
        
        # Update lower arm angle
        if shoulder_pos and elbow_pos:
            lower_arm_angle = calculate_angle(shoulder_pos, elbow_pos)
            self.lower_arm.rotation_z = lower_arm_angle
        
        # Update upper arm angle
        if elbow_pos and wrist_pos:
            upper_arm_angle = calculate_angle(elbow_pos, wrist_pos)
            self.upper_arm.rotation_z = upper_arm_angle
        
        # Update wrist rotation
        self.wrist_joint.rotation_z = hand_angle * 0.5
        self.wrist_joint.rotation_x = hand_angle * 0.3
    
    def fire_repulsor(self):
        # Activate repulsor
        self.repulsor.color = color.rgba(0, 1, 1, 0.9)
        self.repulsor_light.visible = True
        self.repulsor_beam.visible = True
        
        # Create particles
        for _ in range(10):
            particle = Entity(
                model='sphere',
                color=color.rgba(0, 1, 1, 0.7),
                scale=0.05,
                position=(
                    random.uniform(-0.1, 0.1),
                    random.uniform(-0.1, 0.1),
                    random.uniform(0.3, 0.5)
                ),
                parent=self.hand
            )
            self.particles.append(particle)
            
            # Animate particle
            particle.animate_scale((0, 0, 0), duration=1, curve=curve.linear)
            particle.animate_color(color.rgba(0, 0.5, 0.5, 0), duration=1, curve=curve.linear)
            destroy(particle, delay=1)
        
        # Reset repulsor after delay
        invoke(self.reset_repulsor, delay=1)
    
    def reset_repulsor(self):
        self.repulsor.color = color.rgba(0, 0.8, 0.8, 0.7)
        self.repulsor_light.visible = False
        self.repulsor_beam.visible = False
        self.particles = []

    def fist_animation(self):
        # Close fingers to form a fist
        for finger in self.fingers:
            finger.animate_position(
                (finger.position.x, finger.position.y - 0.1, finger.position.z - 0.2),
                duration=0.3
            )
            finger.animate_rotation(
                (45, 0, 0),
                duration=0.3
            )
        
        # Move thumb
        self.thumb.animate_position(
            (self.thumb.position.x + 0.1, self.thumb.position.y, self.thumb.position.z + 0.1),
            duration=0.3
        )
        self.thumb.animate_rotation(
            (0, 45, 45),
            duration=0.3
        )
        
        # Reset after delay
        invoke(self.reset_hand, delay=0.8)
    
    def reset_hand(self):
        # Reset fingers to open position
        for i, finger in enumerate(self.fingers):
            finger.animate_position(
                (-0.15 + i*0.15, 0.2, 0.25),
                duration=0.3
            )
            finger.animate_rotation(
                (0, 0, 0),
                duration=0.3
            )
        
        # Reset thumb
        self.thumb.animate_position(
            (-0.25, 0.1, 0),
            duration=0.3
        )
        self.thumb.animate_rotation(
            (0, 0, 45),
            duration=0.3
        )

    def power_surge_animation(self):
        # Create energy effect along the arm
        glow_points = [self.base, self.rotating_joint, self.lower_arm, 
                       self.mid_joint, self.upper_arm, self.wrist_joint, self.hand]
        
        # Original colors
        original_colors = [entity.color for entity in glow_points]
        
        # Animate power surge
        for i, entity in enumerate(glow_points):
            delay_time = i * 0.1
            invoke(lambda e=entity: setattr(e, 'color', color.yellow), delay=delay_time)
            invoke(lambda e=entity, oc=original_colors[i]: setattr(e, 'color', oc), delay=delay_time + 0.3)
            
            # Create energy particles
            if i > 0:
                invoke(lambda e=entity: self.create_energy_particles(e), delay=delay_time)
    
    def create_energy_particles(self, parent_entity):
        for _ in range(5):
            particle = Entity(
                model='sphere',
                color=color.rgba(1, 1, 0, 0.8),
                scale=0.05,
                position=(
                    random.uniform(-0.2, 0.2),
                    random.uniform(-0.2, 0.2),
                    random.uniform(-0.2, 0.2)
                ),
                parent=parent_entity
            )
            
            # Animate particle
            particle.animate_scale((0, 0, 0), duration=0.5, curve=curve.linear)
            particle.animate_color(color.rgba(1, 1, 0, 0), duration=0.5, curve=curve.linear)
            destroy(particle, delay=0.5)

# Utility function to calculate angle between points
def calculate_angle(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle

# Create the robotic arm
robotic_arm = RoboticArm()

# Add UI elements
instruction_text = Text(
    text="Controls:\nR - Repulsor\nF - Fist\nP - Power Surge\nESC - Exit",
    position=(-0.7, 0.4),
    scale=1.5,
    color=color.white
)

status_text = Text(
    text="Status: Ready",
    position=(-0.7, 0.3),
    scale=1.5,
    color=color.green
)

# MediaPipe Hand tracking setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open the webcam
cap = cv2.VideoCapture(0)

# Initialize previous hand position for smoothing
prev_wrist_pos = [0, 0, 0]
prev_elbow_pos = [0, 0, 0]
prev_shoulder_pos = [0, 0, 0]
prev_hand_angle = 0

smoothing_factor = 0.8  # Higher value means more smoothing

# Input handling
def input(key):
    if key == 'r':
        status_text.text = "Status: Firing Repulsor"
        robotic_arm.fire_repulsor()
        invoke(lambda: setattr(status_text, 'text', "Status: Ready"), delay=1.5)
    
    elif key == 'f':
        status_text.text = "Status: Fist Mode"
        robotic_arm.fist_animation()
        invoke(lambda: setattr(status_text, 'text', "Status: Ready"), delay=1)
    
    elif key == 'p':
        status_text.text = "Status: Power Surge"
        robotic_arm.power_surge_animation()
        invoke(lambda: setattr(status_text, 'text', "Status: Ready"), delay=1.5)
    
    elif key == 'escape':
        # Clean up and exit
        cap.release()
        cv2.destroyAllWindows()
        application.quit()

# Process hand tracking in a separate thread for better performance
def process_hand_tracking():
    global prev_wrist_pos, prev_elbow_pos, prev_shoulder_pos, prev_hand_angle
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to get image from camera")
            continue
        
        # Flip the image horizontally for a mirror effect
        img = cv2.flip(img, 1)
        
        # Convert the BGR image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = hands.process(img_rgb)
        
        wrist_pos = prev_wrist_pos
        elbow_pos = prev_elbow_pos
        shoulder_pos = prev_shoulder_pos
        hand_angle = prev_hand_angle
        
        # If hands are detected
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get wrist position
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            wrist_pos = [
                (wrist.x - 0.5) * 2,  # Map from 0-1 to -1 to 1
                (wrist.y - 0.5) * -2,  # Map from 0-1 to 1 to -1 and flip y-axis
                wrist.z
            ]
            
            # Calculate an approximate elbow position based on wrist and middle finger MCP
            middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            # Elbow is positioned further along the direction from middle_mcp to wrist
            direction_x = wrist.x - middle_mcp.x
            direction_y = wrist.y - middle_mcp.y
            # Scale by a factor to simulate distance to elbow
            elbow_x = wrist.x + direction_x * 3
            elbow_y = wrist.y + direction_y * 3
            
            elbow_pos = [
                (elbow_x - 0.5) * 2,
                (elbow_y - 0.5) * -2,
                wrist.z * 1.5
            ]
            
            # Approximate shoulder position further along the same direction
            shoulder_x = elbow_x + direction_x * 2
            shoulder_y = elbow_y + direction_y * 2
            
            shoulder_pos = [
                (shoulder_x - 0.5) * 2,
                (shoulder_y - 0.5) * -2,
                wrist.z * 2
            ]
            
            # Calculate hand angle based on index finger orientation
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            
            dx = index_tip.x - index_mcp.x
            dy = index_tip.y - index_mcp.y
            hand_angle = math.degrees(math.atan2(dy, dx))
            
            # Detect gestures for special moves
            # Fist detection (thumb close to pinky tip)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            
            dist_thumb_pinky = np.sqrt(
                (thumb_tip.x - pinky_tip.x)**2 + 
                (thumb_tip.y - pinky_tip.y)**2 + 
                (thumb_tip.z - pinky_tip.z)**2
            )
            
            # Repulsor gesture (palm facing camera)
            palm_normal_z = middle_mcp.z - wrist.z
            
            # If palm is facing the camera and fingers are extended
            if palm_normal_z < -0.1 and dist_thumb_pinky > 0.2:
                # Trigger repulsor with reduced frequency to avoid spamming
                if time.time() % 2 < 0.1:
                    invoke(robotic_arm.fire_repulsor)
            
            # If making a fist
            elif dist_thumb_pinky < 0.1:
                # Trigger fist animation with reduced frequency
                if time.time() % 2 < 0.1:
                    invoke(robotic_arm.fist_animation)
            
            # Debug visualization on the camera feed
            # Draw landmarks
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
        # Apply smoothing to hand positions
        wrist_pos = [
            prev_wrist_pos[0] * smoothing_factor + wrist_pos[0] * (1 - smoothing_factor),
            prev_wrist_pos[1] * smoothing_factor + wrist_pos[1] * (1 - smoothing_factor),
            prev_wrist_pos[2] * smoothing_factor + wrist_pos[2] * (1 - smoothing_factor)
        ]
        
        elbow_pos = [
            prev_elbow_pos[0] * smoothing_factor + elbow_pos[0] * (1 - smoothing_factor),
            prev_elbow_pos[1] * smoothing_factor + elbow_pos[1] * (1 - smoothing_factor),
            prev_elbow_pos[2] * smoothing_factor + elbow_pos[2] * (1 - smoothing_factor)
        ]
        
        shoulder_pos = [
            prev_shoulder_pos[0] * smoothing_factor + shoulder_pos[0] * (1 - smoothing_factor),
            prev_shoulder_pos[1] * smoothing_factor + shoulder_pos[1] * (1 - smoothing_factor),
            prev_shoulder_pos[2] * smoothing_factor + shoulder_pos[2] * (1 - smoothing_factor)
        ]
        
        hand_angle = prev_hand_angle * smoothing_factor + hand_angle * (1 - smoothing_factor)
        
        # Save current positions for next frame smoothing
        prev_wrist_pos = wrist_pos
        prev_elbow_pos = elbow_pos
        prev_shoulder_pos = shoulder_pos
        prev_hand_angle = hand_angle
        
        # Display the image
        cv2.imshow("Hand Tracking", img)
        cv2.waitKey(1)
        
        # Sleep a bit to reduce CPU usage
        time.sleep(0.01)

# Start hand tracking in a new thread
import threading
hand_tracking_thread = threading.Thread(target=process_hand_tracking)
hand_tracking_thread.daemon = True
hand_tracking_thread.start()

# Main Ursina update loop
def update():
    # Update robotic arm position based on hand tracking
    robotic_arm.update_position(
        prev_wrist_pos,
        prev_elbow_pos,
        prev_shoulder_pos,
        prev_hand_angle
    )

# Run the application
app.run()

# Cleanup
cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
import math
from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
import time
import random

# Initialize Ursina application
app = Ursina(borderless=False)
window.size = (1280, 720)
window.title = "Advanced Robotic Arm Simulation"

# Set up camera and lighting
camera.position = (0, 2, -8)
camera.rotation = (15, 0, 0)
DirectionalLight(parent=scene, y=2, z=3, shadows=True, rotation=(45, -45, 45))
AmbientLight(parent=scene, color=color.rgba(100, 100, 100, 0.1))

# Create environment
environment = Entity(
    model='sphere',
    texture='textures/skybox',
    scale=500,
    double_sided=True
)

# Create a platform for the robotic arm
platform = Entity(
    model='cube',
    color=color.dark_gray,
    scale=(3, 0.2, 3),
    position=(0, 0, 0),
    texture='white_cube',
    texture_scale=(3, 3),
    collider='box'
)

# Create robotic arm components
class RoboticArm:
    def __init__(self):
        # Base
        self.base = Entity(
            model='cylinder',
            color=color.dark_gray,
            scale=(1, 0.2, 1),
            position=(0, 0.2, 0),
            texture='white_cube'
        )
        
        # Rotating joint
        self.rotating_joint = Entity(
            model='sphere',
            color=color.gray,
            scale=0.5,
            position=(0, 0.5, 0),
            parent=self.base
        )
        
        # Lower arm
        self.lower_arm = Entity(
            model='cube',
            color=color.light_gray,
            scale=(0.3, 1.5, 0.3),
            position=(0, 0.8, 0),
            parent=self.rotating_joint
        )
        
        # Mid joint
        self.mid_joint = Entity(
            model='sphere',
            color=color.gray,
            scale=0.4,
            position=(0, 1.5, 0),
            parent=self.lower_arm
        )
        
        # Upper arm
        self.upper_arm = Entity(
            model='cube',
            color=color.light_gray,
            scale=(0.25, 1.3, 0.25),
            position=(0, 0.7, 0),
            parent=self.mid_joint
        )
        
        # Wrist joint
        self.wrist_joint = Entity(
            model='sphere',
            color=color.gray,
            scale=0.3,
            position=(0, 1.3, 0),
            parent=self.upper_arm
        )
        
        # Hand
        self.hand = Entity(
            model='cube',
            color=color.white,
            scale=(0.4, 0.2, 0.6),
            position=(0, 0.3, 0),
            parent=self.wrist_joint
        )
        
        # Fingers
        self.fingers = []
        for i in range(3):
            finger = Entity(
                model='cube',
                color=color.white,
                scale=(0.08, 0.4, 0.08),
                position=(-0.15 + i*0.15, 0.2, 0.25),
                parent=self.hand
            )
            self.fingers.append(finger)
        
        # Thumb
        self.thumb = Entity(
            model='cube',
            color=color.white,
            scale=(0.08, 0.3, 0.08),
            position=(-0.25, 0.1, 0),
            rotation=(0, 0, 45),
            parent=self.hand
        )
        
        # Add details to the arm
        self.details = []
        # Hydraulic cylinders
        hydraulic1 = Entity(
            model='cylinder',
            color=color.dark_gray,
            scale=(0.05, 1.2, 0.05),
            position=(0.15, 0.6, 0.15),
            rotation=(0, 0, 15),
            parent=self.lower_arm
        )
        self.details.append(hydraulic1)
        
        hydraulic2 = Entity(
            model='cylinder',
            color=color.dark_gray,
            scale=(0.05, 1, 0.05),
            position=(0.15, 0.5, 0.15),
            rotation=(0, 0, 15),
            parent=self.upper_arm
        )
        self.details.append(hydraulic2)
        
        # Wires
        for i in range(5):
            wire = Entity(
                model='cylinder',
                color=color.random_color(),
                scale=(0.02, 3, 0.02),
                position=(-0.2 + i*0.02, 0.5, -0.2),
                parent=self.lower_arm
            )
            self.details.append(wire)
        
        # Palm repulsor
        self.repulsor = Entity(
            model='sphere',
            color=color.rgba(0, 0.8, 0.8, 0.7),
            scale=0.15,
            position=(0, 0, 0.31),
            parent=self.hand
        )
        
        # Repulsor light
        self.repulsor_light = PointLight(
            parent=self.repulsor,
            color=color.cyan,
            intensity=0.5
        )
        self.repulsor_light.visible = False
        
        # Repulsor beam
        self.repulsor_beam = Entity(
            model='cylinder',
            color=color.rgba(0, 0.8, 0.8, 0.5),
            scale=(0.1, 15, 0.1),
            position=(0, 7.5, 0),
            rotation=(90, 0, 0),
            parent=self.repulsor
        )
        self.repulsor_beam.visible = False
        
        # Particles for the repulsor
        self.particles = []
        
    def update_position(self, wrist_pos, elbow_pos, shoulder_pos, hand_angle):
        # Scale and adjust positions to fit the 3D space
        scale_factor = 5
        
        # Update base rotation to track hand movement on horizontal plane
        target_x = wrist_pos[0] * scale_factor
        target_z = wrist_pos[2] * scale_factor
        
        base_rotation = math.degrees(math.atan2(target_x, target_z))
        self.rotating_joint.rotation_y = base_rotation
        
        # Update lower arm angle
        if shoulder_pos and elbow_pos:
            lower_arm_angle = calculate_angle(shoulder_pos, elbow_pos)
            self.lower_arm.rotation_z = lower_arm_angle
        
        # Update upper arm angle
        if elbow_pos and wrist_pos:
            upper_arm_angle = calculate_angle(elbow_pos, wrist_pos)
            self.upper_arm.rotation_z = upper_arm_angle
        
        # Update wrist rotation
        self.wrist_joint.rotation_z = hand_angle * 0.5
        self.wrist_joint.rotation_x = hand_angle * 0.3
    
    def fire_repulsor(self):
        # Activate repulsor
        self.repulsor.color = color.rgba(0, 1, 1, 0.9)
        self.repulsor_light.visible = True
        self.repulsor_beam.visible = True
        
        # Create particles
        for _ in range(10):
            particle = Entity(
                model='sphere',
                color=color.rgba(0, 1, 1, 0.7),
                scale=0.05,
                position=(
                    random.uniform(-0.1, 0.1),
                    random.uniform(-0.1, 0.1),
                    random.uniform(0.3, 0.5)
                ),
                parent=self.hand
            )
            self.particles.append(particle)
            
            # Animate particle
            particle.animate_scale((0, 0, 0), duration=1, curve=curve.linear)
            particle.animate_color(color.rgba(0, 0.5, 0.5, 0), duration=1, curve=curve.linear)
            destroy(particle, delay=1)
        
        # Reset repulsor after delay
        invoke(self.reset_repulsor, delay=1)
    
    def reset_repulsor(self):
        self.repulsor.color = color.rgba(0, 0.8, 0.8, 0.7)
        self.repulsor_light.visible = False
        self.repulsor_beam.visible = False
        self.particles = []

    def fist_animation(self):
        # Close fingers to form a fist
        for finger in self.fingers:
            finger.animate_position(
                (finger.position.x, finger.position.y - 0.1, finger.position.z - 0.2),
                duration=0.3
            )
            finger.animate_rotation(
                (45, 0, 0),
                duration=0.3
            )
        
        # Move thumb
        self.thumb.animate_position(
            (self.thumb.position.x + 0.1, self.thumb.position.y, self.thumb.position.z + 0.1),
            duration=0.3
        )
        self.thumb.animate_rotation(
            (0, 45, 45),
            duration=0.3
        )
        
        # Reset after delay
        invoke(self.reset_hand, delay=0.8)
    
    def reset_hand(self):
        # Reset fingers to open position
        for i, finger in enumerate(self.fingers):
            finger.animate_position(
                (-0.15 + i*0.15, 0.2, 0.25),
                duration=0.3
            )
            finger.animate_rotation(
                (0, 0, 0),
                duration=0.3
            )
        
        # Reset thumb
        self.thumb.animate_position(
            (-0.25, 0.1, 0),
            duration=0.3
        )
        self.thumb.animate_rotation(
            (0, 0, 45),
            duration=0.3
        )

    def power_surge_animation(self):
        # Create energy effect along the arm
        glow_points = [self.base, self.rotating_joint, self.lower_arm, 
                       self.mid_joint, self.upper_arm, self.wrist_joint, self.hand]
        
        # Original colors
        original_colors = [entity.color for entity in glow_points]
        
        # Animate power surge
        for i, entity in enumerate(glow_points):
            delay_time = i * 0.1
            invoke(lambda e=entity: setattr(e, 'color', color.yellow), delay=delay_time)
            invoke(lambda e=entity, oc=original_colors[i]: setattr(e, 'color', oc), delay=delay_time + 0.3)
            
            # Create energy particles
            if i > 0:
                invoke(lambda e=entity: self.create_energy_particles(e), delay=delay_time)
    
    def create_energy_particles(self, parent_entity):
        for _ in range(5):
            particle = Entity(
                model='sphere',
                color=color.rgba(1, 1, 0, 0.8),
                scale=0.05,
                position=(
                    random.uniform(-0.2, 0.2),
                    random.uniform(-0.2, 0.2),
                    random.uniform(-0.2, 0.2)
                ),
                parent=parent_entity
            )
            
            # Animate particle
            particle.animate_scale((0, 0, 0), duration=0.5, curve=curve.linear)
            particle.animate_color(color.rgba(1, 1, 0, 0), duration=0.5, curve=curve.linear)
            destroy(particle, delay=0.5)

# Utility function to calculate angle between points
def calculate_angle(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle

# Create the robotic arm
robotic_arm = RoboticArm()

# Add UI elements
instruction_text = Text(
    text="Controls:\nR - Repulsor\nF - Fist\nP - Power Surge\nESC - Exit",
    position=(-0.7, 0.4),
    scale=1.5,
    color=color.white
)

status_text = Text(
    text="Status: Ready",
    position=(-0.7, 0.3),
    scale=1.5,
    color=color.green
)

# MediaPipe Hand tracking setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open the webcam
cap = cv2.VideoCapture(0)

# Initialize previous hand position for smoothing
prev_wrist_pos = [0, 0, 0]
prev_elbow_pos = [0, 0, 0]
prev_shoulder_pos = [0, 0, 0]
prev_hand_angle = 0

smoothing_factor = 0.8  # Higher value means more smoothing

# Input handling
def input(key):
    if key == 'r':
        status_text.text = "Status: Firing Repulsor"
        robotic_arm.fire_repulsor()
        invoke(lambda: setattr(status_text, 'text', "Status: Ready"), delay=1.5)
    
    elif key == 'f':
        status_text.text = "Status: Fist Mode"
        robotic_arm.fist_animation()
        invoke(lambda: setattr(status_text, 'text', "Status: Ready"), delay=1)
    
    elif key == 'p':
        status_text.text = "Status: Power Surge"
        robotic_arm.power_surge_animation()
        invoke(lambda: setattr(status_text, 'text', "Status: Ready"), delay=1.5)
    
    elif key == 'escape':
        # Clean up and exit
        cap.release()
        cv2.destroyAllWindows()
        application.quit()

# Process hand tracking in a separate thread for better performance
def process_hand_tracking():
    global prev_wrist_pos, prev_elbow_pos, prev_shoulder_pos, prev_hand_angle
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to get image from camera")
            continue
        
        # Flip the image horizontally for a mirror effect
        img = cv2.flip(img, 1)
        
        # Convert the BGR image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = hands.process(img_rgb)
        
        wrist_pos = prev_wrist_pos
        elbow_pos = prev_elbow_pos
        shoulder_pos = prev_shoulder_pos
        hand_angle = prev_hand_angle
        
        # If hands are detected
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get wrist position
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            wrist_pos = [
                (wrist.x - 0.5) * 2,  # Map from 0-1 to -1 to 1
                (wrist.y - 0.5) * -2,  # Map from 0-1 to 1 to -1 and flip y-axis
                wrist.z
            ]
            
            # Calculate an approximate elbow position based on wrist and middle finger MCP
            middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            # Elbow is positioned further along the direction from middle_mcp to wrist
            direction_x = wrist.x - middle_mcp.x
            direction_y = wrist.y - middle_mcp.y
            # Scale by a factor to simulate distance to elbow
            elbow_x = wrist.x + direction_x * 3
            elbow_y = wrist.y + direction_y * 3
            
            elbow_pos = [
                (elbow_x - 0.5) * 2,
                (elbow_y - 0.5) * -2,
                wrist.z * 1.5
            ]
            
            # Approximate shoulder position further along the same direction
            shoulder_x = elbow_x + direction_x * 2
            shoulder_y = elbow_y + direction_y * 2
            
            shoulder_pos = [
                (shoulder_x - 0.5) * 2,
                (shoulder_y - 0.5) * -2,
                wrist.z * 2
            ]
            
            # Calculate hand angle based on index finger orientation
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            
            dx = index_tip.x - index_mcp.x
            dy = index_tip.y - index_mcp.y
            hand_angle = math.degrees(math.atan2(dy, dx))
            
            # Detect gestures for special moves
            # Fist detection (thumb close to pinky tip)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            
            dist_thumb_pinky = np.sqrt(
                (thumb_tip.x - pinky_tip.x)**2 + 
                (thumb_tip.y - pinky_tip.y)**2 + 
                (thumb_tip.z - pinky_tip.z)**2
            )
            
            # Repulsor gesture (palm facing camera)
            palm_normal_z = middle_mcp.z - wrist.z
            
            # If palm is facing the camera and fingers are extended
            if palm_normal_z < -0.1 and dist_thumb_pinky > 0.2:
                # Trigger repulsor with reduced frequency to avoid spamming
                if time.time() % 2 < 0.1:
                    invoke(robotic_arm.fire_repulsor)
            
            # If making a fist
            elif dist_thumb_pinky < 0.1:
                # Trigger fist animation with reduced frequency
                if time.time() % 2 < 0.1:
                    invoke(robotic_arm.fist_animation)
            
            # Debug visualization on the camera feed
            # Draw landmarks
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
        # Apply smoothing to hand positions
        wrist_pos = [
            prev_wrist_pos[0] * smoothing_factor + wrist_pos[0] * (1 - smoothing_factor),
            prev_wrist_pos[1] * smoothing_factor + wrist_pos[1] * (1 - smoothing_factor),
            prev_wrist_pos[2] * smoothing_factor + wrist_pos[2] * (1 - smoothing_factor)
        ]
        
        elbow_pos = [
            prev_elbow_pos[0] * smoothing_factor + elbow_pos[0] * (1 - smoothing_factor),
            prev_elbow_pos[1] * smoothing_factor + elbow_pos[1] * (1 - smoothing_factor),
            prev_elbow_pos[2] * smoothing_factor + elbow_pos[2] * (1 - smoothing_factor)
        ]
        
        shoulder_pos = [
            prev_shoulder_pos[0] * smoothing_factor + shoulder_pos[0] * (1 - smoothing_factor),
            prev_shoulder_pos[1] * smoothing_factor + shoulder_pos[1] * (1 - smoothing_factor),
            prev_shoulder_pos[2] * smoothing_factor + shoulder_pos[2] * (1 - smoothing_factor)
        ]
        
        hand_angle = prev_hand_angle * smoothing_factor + hand_angle * (1 - smoothing_factor)
        
        # Save current positions for next frame smoothing
        prev_wrist_pos = wrist_pos
        prev_elbow_pos = elbow_pos
        prev_shoulder_pos = shoulder_pos
        prev_hand_angle = hand_angle
        
        # Display the image
        cv2.imshow("Hand Tracking", img)
        cv2.waitKey(1)
        
        # Sleep a bit to reduce CPU usage
        time.sleep(0.01)

# Start hand tracking in a new thread
import threading
hand_tracking_thread = threading.Thread(target=process_hand_tracking)
hand_tracking_thread.daemon = True
hand_tracking_thread.start()

# Main Ursina update loop
def update():
    # Update robotic arm position based on hand tracking
    robotic_arm.update_position(
        prev_wrist_pos,
        prev_elbow_pos,
        prev_shoulder_pos,
        prev_hand_angle
    )

# Run the application
app.run()

# Cleanup
cap.release()
cv2.destroyAllWindows()