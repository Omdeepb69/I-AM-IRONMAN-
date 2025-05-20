import cv2
import mediapipe as mp
import numpy as np
import time
import math
from ursina import *
from ursina.shaders import lit_with_shadows_shader
from ursina.texture_importer import load_texture

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class RepulsorBlast(Entity):
    def __init__(self, position, direction):
        super().__init__(
            model='sphere',
            color=color.rgba(0, 200, 255, 200),
            scale=0.2,
            position=position,
            collider='sphere',
            add_to_scene_entities=False
        )
        self.direction = direction.normalized()
        self.speed = 15
        self.lifetime = 1.5
        self.born = time.time()
        
        # Glow effect
        self.glow = Entity(
            parent=self,
            model='sphere',
            color=color.rgba(0, 150, 255, 100),
            scale=1.5,
            double_sided=True,
            add_to_scene_entities=False
        )
        
        # Trail particles
        self.particles = []
        for _ in range(30):
            p = Entity(
                model='sphere',
                color=color.rgba(100, 200, 255, 150),
                scale=random.uniform(0.05, 0.1),
                position=position,
                add_to_scene_entities=False
            )
            self.particles.append(p)

    def update(self):
        if not self.enabled:
            return
            
        # Move blast forward
        self.position += self.direction * self.speed * time.dt
        
        # Update glow
        self.glow.position = self.position
        self.glow.scale += Vec3(0.8, 0.8, 0.8) * time.dt
        
        # Update particles
        for p in self.particles:
            if random.random() < 0.3:
                p.position = self.position
            p.position += Vec3(
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                random.uniform(-1, 1)
            ) * time.dt * 3
            p.scale -= Vec3(0.03, 0.03, 0.03) * time.dt
        
        # Destroy after lifetime
        if time.time() - self.born > self.lifetime:
            self.enabled = False
            self.glow.enabled = False
            for p in self.particles:
                p.enabled = False

class IronManArm(Entity):
    def __init__(self):
        super().__init__()
        
        # Load metal textures
        self.red_metal = load_texture('red_metal')
        self.gold_metal = load_texture('gold_metal')
        self.blue_energy = load_texture('blue_energy')
        
        # Create arm hierarchy with proper transforms
        self.upper_arm = Entity(
            parent=self,
            model='cube',
            texture=self.red_metal,
            color=color.rgb(200, 50, 50),
            scale=(0.5, 1.2, 0.5),
            position=(0, 0, 0),
            rotation=(0, 0, 0),
            shader=lit_with_shadows_shader
        )
        
        # Shoulder joint
        self.shoulder = Entity(
            parent=self.upper_arm,
            model='sphere',
            texture=self.gold_metal,
            color=color.gold,
            scale=0.3,
            position=(0, 0.6, 0),
            shader=lit_with_shadows_shader
        )
        
        # Elbow joint
        self.elbow = Entity(
            parent=self.upper_arm,
            model='sphere',
            texture=self.gold_metal,
            color=color.gold,
            scale=0.25,
            position=(0, -0.6, 0),
            shader=lit_with_shadows_shader
        )
        
        self.forearm = Entity(
            parent=self.elbow,
            model='cube',
            texture=self.red_metal,
            color=color.rgb(200, 50, 50),
            scale=(0.45, 1.1, 0.45),
            position=(0, -0.55, 0),
            rotation=(0, 0, 0),
            shader=lit_with_shadows_shader
        )
        
        # Wrist joint
        self.wrist = Entity(
            parent=self.forearm,
            model='sphere',
            texture=self.gold_metal,
            color=color.gold,
            scale=0.2,
            position=(0, -0.55, 0),
            shader=lit_with_shadows_shader
        )
        
        self.hand = Entity(
            parent=self.wrist,
            model='cube',
            texture=self.red_metal,
            color=color.rgb(200, 50, 50),
            scale=(0.4, 0.4, 0.4),
            position=(0, -0.2, 0),
            rotation=(0, 0, 0),
            shader=lit_with_shadows_shader
        )
        
        # Palm with repulsor
        self.palm = Entity(
            parent=self.hand,
            model='cube',
            texture=self.gold_metal,
            color=color.gold,
            scale=(0.35, 0.2, 0.5),
            position=(0, 0, 0.3),
            rotation=(0, 0, 0),
            shader=lit_with_shadows_shader
        )
        
        # Fingers with proper hierarchy
        self.create_fingers()
        
        # Repulsor system
        self.setup_repulsor()
        
        # Hand tracking
        self.setup_hand_tracking()
        
        # Animation state
        self.setup_animation_state()

    def create_fingers(self):
        # Thumb
        self.thumb_base = Entity(
            parent=self.hand,
            model='cube',
            texture=self.red_metal,
            color=color.rgb(200, 50, 50),
            scale=(0.15, 0.15, 0.15),
            position=(-0.2, 0, 0.1),
            rotation=(0, 0, -20),
            shader=lit_with_shadows_shader
        )
        
        self.thumb_mid = Entity(
            parent=self.thumb_base,
            model='cube',
            texture=self.red_metal,
            scale=(0.9, 0.9, 0.9),
            position=(0, 0, 0.15),
            rotation=(0, 0, 0),
            shader=lit_with_shadows_shader
        )
        
        self.thumb_tip = Entity(
            parent=self.thumb_mid,
            model='cube',
            texture=self.red_metal,
            scale=(0.9, 0.9, 0.9),
            position=(0, 0, 0.15),
            rotation=(0, 0, 0),
            shader=lit_with_shadows_shader
        )
        
        # Other fingers (index, middle, ring, pinky)
        finger_positions = [
            (0.1, -0.1, 0.2),  # index
            (0.0, -0.15, 0.2),  # middle
            (-0.1, -0.1, 0.2),  # ring
            (-0.2, -0.05, 0.2)  # pinky
        ]
        
        self.fingers = []
        for i, pos in enumerate(finger_positions):
            base = Entity(
                parent=self.hand,
                model='cube',
                texture=self.red_metal,
                scale=(0.1, 0.1, 0.1),
                position=pos,
                rotation=(0, 0, 0),
                shader=lit_with_shadows_shader
            )
            
            mid = Entity(
                parent=base,
                model='cube',
                texture=self.red_metal,
                scale=(0.9, 0.9, 0.9),
                position=(0, 0, 0.1),
                rotation=(0, 0, 0),
                shader=lit_with_shadows_shader
            )
            
            tip = Entity(
                parent=mid,
                model='cube',
                texture=self.red_metal,
                scale=(0.9, 0.9, 0.9),
                position=(0, 0, 0.1),
                rotation=(0, 0, 0),
                shader=lit_with_shadows_shader
            )
            
            self.fingers.append((base, mid, tip))

    def setup_repulsor(self):
        # Repulsor core
        self.repulsor = Entity(
            parent=self.palm,
            model='circle',
            texture=self.blue_energy,
            color=color.rgba(0, 150, 255, 200),
            scale=0.3,
            position=(0, 0, 0.3),
            rotation=(90, 0, 0),
            shader=lit_with_shadows_shader
        )
        
        # Energy glow
        self.repulsor_glow = Entity(
            parent=self.repulsor,
            model='circle',
            color=color.rgba(0, 100, 255, 100),
            scale=1.5,
            double_sided=True,
            shader=lit_with_shadows_shader
        )
        
        # Lighting
        self.repulsor_light = PointLight(
            parent=self.repulsor,
            color=color.cyan,
            shadows=False,
            position=(0, 0, 0)
        )
        
        # State
        self.repulsor_charging = False
        self.repulsor_charge_time = 0
        self.repulsor_cooldown = 0
        self.energy_level = 0

    def setup_hand_tracking(self):
        self.hand_landmarks = None
        self.cap = cv2.VideoCapture(0)
        self.hands = mp_hands.Hands(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            max_num_hands=1
        )

    def setup_animation_state(self):
        # Target rotations for smooth animation
        self.target_rotations = {
            'upper_arm': Vec3(0, 0, 0),
            'elbow': Vec3(0, 0, 0),
            'wrist': Vec3(0, 0, 0),
            'hand': Vec3(0, 0, 0),
            'thumb': [Vec3(0, 0, -20), Vec3(0, 0, 0), Vec3(0, 0, 0)],
            'fingers': [
                [Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0)] for _ in range(4)
            ]
        }
        
        # Current rotations for interpolation
        self.current_rotations = {
            'upper_arm': Vec3(0, 0, 0),
            'elbow': Vec3(0, 0, 0),
            'wrist': Vec3(0, 0, 0),
            'hand': Vec3(0, 0, 0),
            'thumb': [Vec3(0, 0, -20), Vec3(0, 0, 0), Vec3(0, 0, 0)],
            'fingers': [
                [Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0)] for _ in range(4)
            ]
        }
        
        self.rotation_speed = 30  # Slower for smoother movement

    def update(self):
        # Process hand tracking
        self.process_hand_input()
        
        # Animate arm
        self.animate_arm()
        
        # Handle repulsor
        self.handle_repulsor()

    def process_hand_input(self):
        success, image = self.cap.read()
        if not success:
            return
        
        # Process image
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True
        
        if results.multi_hand_landmarks:
            self.hand_landmarks = results.multi_hand_landmarks[0]
            self.update_arm_pose()

    def update_arm_pose(self):
        landmarks = self.hand_landmarks.landmark
        
        # Get key points
        wrist = landmarks[0]
        elbow = landmarks[13]
        middle_tip = landmarks[12]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        pinky_tip = landmarks[20]
        
        # Calculate arm rotations with constraints
        upper_arm_rot_x = clamp((elbow.y - wrist.y) * 90, -90, 90)
        upper_arm_rot_y = clamp((elbow.x - wrist.x) * -90, -90, 90)
        
        forearm_rot_x = clamp((middle_tip.y - wrist.y) * 90 - upper_arm_rot_x, -90, 90)
        forearm_rot_y = clamp((middle_tip.x - wrist.x) * -90 - upper_arm_rot_y, -90, 90)
        
        hand_rot_y = clamp((pinky_tip.x - index_tip.x) * -45, -45, 45)
        hand_rot_z = clamp((wrist.y - middle_tip.y) * -45, -45, 45)
        
        # Update target rotations
        self.target_rotations['upper_arm'] = Vec3(upper_arm_rot_x, upper_arm_rot_y, 0)
        self.target_rotations['elbow'] = Vec3(forearm_rot_x, forearm_rot_y, 0)
        self.target_rotations['wrist'] = Vec3(0, hand_rot_y, hand_rot_z)
        
        # Update finger rotations
        self.update_finger_rotations()

    def update_finger_rotations(self):
        landmarks = self.hand_landmarks.landmark
        
        # Thumb rotations
        thumb_rot_z = clamp((landmarks[4].x - landmarks[2].x) * -90 - 20, -90, 90)
        self.target_rotations['thumb'][0] = Vec3(0, 0, thumb_rot_z)
        self.target_rotations['thumb'][1] = Vec3(0, 0, thumb_rot_z * 0.7)
        self.target_rotations['thumb'][2] = Vec3(0, 0, thumb_rot_z * 0.5)
        
        # Other fingers
        finger_joints = [
            (8, 6, 5),   # index
            (12, 10, 9), # middle
            (16, 14, 13), # ring
            (20, 18, 17)  # pinky
        ]
        
        for i, (tip, pip, dip) in enumerate(finger_joints):
            bend = clamp((landmarks[tip].y - landmarks[pip].y) * -90, 0, 90)
            self.target_rotations['fingers'][i][0] = Vec3(bend * 0.3, 0, 0)
            self.target_rotations['fingers'][i][1] = Vec3(bend * 0.7, 0, 0)
            self.target_rotations['fingers'][i][2] = Vec3(bend, 0, 0)
        
        # Check for repulsor activation
        self.check_repulsor_activation()

    def check_repulsor_activation(self):
        landmarks = self.hand_landmarks.landmark
        
        # Check if hand is open
        fingers_open = all(
            landmarks[tip].y < landmarks[tip-2].y
            for tip in [8, 12, 16, 20]  # index, middle, ring, pinky tips
        ) and landmarks[4].x > landmarks[2].x  # thumb not extended
        
        if fingers_open:
            if not self.repulsor_charging:
                self.start_repulsor_charge()
            elif time.time() - self.repulsor_charge_time > 2 and self.repulsor_cooldown <= 0:
                self.fire_repulsor()
        else:
            if self.repulsor_charging:
                self.cancel_repulsor_charge()
        
        if self.repulsor_cooldown > 0:
            self.repulsor_cooldown -= time.dt

    def animate_arm(self):
        # Interpolate rotations for smooth movement
        self.animate_part(self.upper_arm, 'upper_arm')
        self.animate_part(self.forearm, 'elbow')
        self.animate_part(self.hand, 'wrist')
        
        # Animate thumb
        for i, part in enumerate([self.thumb_base, self.thumb_mid, self.thumb_tip]):
            self.animate_finger_part(part, 'thumb', i)
        
        # Animate fingers
        for finger_idx, finger_parts in enumerate(self.fingers):
            for part_idx, part in enumerate(finger_parts):
                self.animate_finger_part(part, 'fingers', finger_idx, part_idx)

    def animate_part(self, part, part_name):
        current = self.current_rotations[part_name]
        target = self.target_rotations[part_name]
        
        new_rot = Vec3(
            lerp(current.x, target.x, time.dt * self.rotation_speed),
            lerp(current.y, target.y, time.dt * self.rotation_speed),
            lerp(current.z, target.z, time.dt * self.rotation_speed)
        )
        
        # Safety check for NaN values
        if any(math.isnan(v) for v in new_rot):
            new_rot = Vec3(0, 0, 0)
        
        self.current_rotations[part_name] = new_rot
        part.rotation = new_rot

    def animate_finger_part(self, part, part_type, finger_idx, part_idx=None):
        if part_type == 'thumb':
            current = self.current_rotations[part_type][finger_idx]
            target = self.target_rotations[part_type][finger_idx]
        else:
            current = self.current_rotations[part_type][finger_idx][part_idx]
            target = self.target_rotations[part_type][finger_idx][part_idx]
        
        new_rot = Vec3(
            lerp(current.x, target.x, time.dt * self.rotation_speed * 2),
            lerp(current.y, target.y, time.dt * self.rotation_speed * 2),
            lerp(current.z, target.z, time.dt * self.rotation_speed * 2)
        )
        
        if any(math.isnan(v) for v in new_rot):
            new_rot = Vec3(0, 0, 0)
        
        if part_type == 'thumb':
            self.current_rotations[part_type][finger_idx] = new_rot
        else:
            self.current_rotations[part_type][finger_idx][part_idx] = new_rot
        
        part.rotation = new_rot

    def start_repulsor_charge(self):
        self.repulsor_charging = True
        self.repulsor_charge_time = time.time()
        
        # Visual effects
        self.repulsor.color = color.rgba(0, 200, 255, 255)
        self.repulsor_glow.color = color.rgba(0, 150, 255, 150)
        self.repulsor_light.color = color.rgba(0, 200, 255)

    def cancel_repulsor_charge(self):
        self.repulsor_charging = False
        
        # Reset visual effects
        self.repulsor.color = color.rgba(0, 150, 255, 200)
        self.repulsor_glow.color = color.rgba(0, 100, 255, 100)
        self.repulsor_light.color = color.cyan
        self.energy_level = 0

    def fire_repulsor(self):
        # Create blast
        blast = RepulsorBlast(
            position=self.repulsor.world_position,
            direction=self.palm.forward
        )
        
        # Cooldown
        self.repulsor_cooldown = 1.0
        self.repulsor_charging = False
        
        # Reset charge
        invoke(self.cancel_repulsor_charge, delay=0.5)

    def handle_repulsor(self):
        if self.repulsor_charging:
            # Update charge level
            self.energy_level = clamp((time.time() - self.repulsor_charge_time) / 2.0, 0, 1)
            
            # Visual feedback
            pulse = math.sin(time.time() * 20) * 0.1 + 1.0
            self.repulsor_glow.scale = 1.5 + self.energy_level * pulse
            self.repulsor_glow.color = color.rgba(
                0, 
                150 + int(105 * self.energy_level), 
                255, 
                100 + int(155 * self.energy_level)
            )

app = Ursina()

# Configure window
window.title = 'Iron Man Arm'
window.borderless = False
window.fullscreen = False
window.exit_button.visible = False
window.fps_counter.enabled = True
window.color = color.black

# Setup camera
camera.position = (0, 0, -6)
camera.rotation = (0, 0, 0)

# Create arm
arm = IronManArm()
arm.position = (0, -1.5, 0)
arm.rotation_y = 180

# Light
DirectionalLight(color=color.white, direction=(1, -1, 1))
AmbientLight(color=color.rgba(100, 100, 100, 0.1))

def update():
    arm.update()

def input(key):
    if key == 'escape':
        application.quit()

app.run()