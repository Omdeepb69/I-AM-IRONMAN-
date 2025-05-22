import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import math
import random

class AccurateIronManHand:
    def __init__(self):
        # Iron Man specific colors - movie accurate
        self.colors = {
            'gold': [1.0, 0.843, 0.0],        # Bright gold
            'red': [0.698, 0.133, 0.133],     # Deep red
            'dark_red': [0.545, 0.0, 0.0],   # Dark red
            'black': [0.1, 0.1, 0.1],        # Near black
            'silver': [0.753, 0.753, 0.753], # Silver details
            'blue_glow': [0.0, 0.749, 1.0],  # Arc reactor blue
            'white': [1.0, 1.0, 1.0],        # White highlights
            'gunmetal': [0.267, 0.267, 0.267] # Dark metallic
        }
        
        # Joint system for realistic movement
        self.joints = {
            'wrist_bend': 0,
            'wrist_twist': 0,
            'thumb': [0, 0, 0],    # 3 joints per finger
            'index': [0, 0, 0],
            'middle': [0, 0, 0],
            'ring': [0, 0, 0],
            'pinky': [0, 0, 0]
        }
        
        self.time = 0
        self.pose_mode = 'open'
        self.repulsor_power = 0.0

    def create_sphere(self, center, radius, resolution=12):
        """Create a sphere for rounded parts"""
        phi = np.linspace(0, np.pi, resolution)
        theta = np.linspace(0, 2*np.pi, resolution)
        phi, theta = np.meshgrid(phi, theta)
        
        x = center[0] + radius * np.sin(phi) * np.cos(theta)
        y = center[1] + radius * np.sin(phi) * np.sin(theta)
        z = center[2] + radius * np.cos(phi)
        
        vertices = []
        faces = []
        
        for i in range(resolution-1):
            for j in range(resolution-1):
                vertices.extend([
                    [x[i,j], y[i,j], z[i,j]],
                    [x[i+1,j], y[i+1,j], z[i+1,j]],
                    [x[i,j+1], y[i,j+1], z[i,j+1]],
                    [x[i+1,j+1], y[i+1,j+1], z[i+1,j+1]]
                ])
                
                base = len(vertices) - 4
                faces.extend([
                    [base, base+1, base+2],
                    [base+1, base+3, base+2]
                ])
        
        return vertices, faces

    def create_rounded_box(self, center, size, corner_radius=0.1):
        """Create Iron Man style rounded rectangular segments"""
        cx, cy, cz = center
        sx, sy, sz = size
        
        vertices = []
        faces = []
        
        # Create main box with rounded corners
        corners = [
            [-sx/2, -sy/2, -sz/2], [sx/2, -sy/2, -sz/2],
            [sx/2, sy/2, -sz/2], [-sx/2, sy/2, -sz/2],
            [-sx/2, -sy/2, sz/2], [sx/2, -sy/2, sz/2],
            [sx/2, sy/2, sz/2], [-sx/2, sy/2, sz/2]
        ]
        
        # Add slight curve to make it more organic
        for i, corner in enumerate(corners):
            x, y, z = corner
            # Add subtle curvature
            curve_factor = 0.95
            if abs(x) > sx/4: x *= curve_factor
            if abs(y) > sy/4: y *= curve_factor
            vertices.append([cx + x, cy + y, cz + z])
        
        # Define faces for the box
        box_faces = [
            [0,1,2,3], [4,7,6,5], [0,4,5,1],  # bottom, top, front
            [2,6,7,3], [0,3,7,4], [1,5,6,2]   # back, left, right
        ]
        
        # Convert quads to triangles
        for face in box_faces:
            faces.extend([[face[0], face[1], face[2]], [face[0], face[2], face[3]]])
        
        return vertices, faces

    def create_finger_segment(self, start_pos, end_pos, thickness, taper=0.9):
        """Create accurate Iron Man finger segment with proper proportions"""
        start = np.array(start_pos)
        end = np.array(end_pos)
        direction = end - start
        length = np.linalg.norm(direction)
        
        if length == 0:
            return [], []
        
        direction = direction / length
        
        # Create perpendicular vectors
        if abs(direction[2]) < 0.9:
            perp1 = np.cross(direction, [0, 0, 1])
        else:
            perp1 = np.cross(direction, [1, 0, 0])
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)
        
        segments = 8
        vertices = []
        faces = []
        
        for i in range(segments + 1):
            t = i / segments
            pos = start + t * direction * length
            radius = thickness * (1 - t * (1 - taper))
            
            # Create octagonal cross-section for Iron Man look
            for j in range(8):
                angle = 2 * np.pi * j / 8
                offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
                # Flatten slightly for more mechanical look
                if j % 2 == 1:
                    offset *= 0.85
                vertices.append(pos + offset)
        
        # Create faces
        for i in range(segments):
            for j in range(8):
                next_j = (j + 1) % 8
                v1 = i * 8 + j
                v2 = i * 8 + next_j
                v3 = (i + 1) * 8 + next_j
                v4 = (i + 1) * 8 + j
                
                faces.extend([[v1, v2, v3], [v1, v3, v4]])
        
        return vertices, faces

    def create_knuckle_joint(self, position, size):
        """Create Iron Man style knuckle joints"""
        vertices, faces = self.create_rounded_box(position, [size*1.2, size*1.2, size*0.8])
        
        # Add joint details
        detail_vertices, detail_faces = self.create_sphere(position, size*0.3, 6)
        
        offset = len(vertices)
        for face in detail_faces:
            faces.append([face[0] + offset, face[1] + offset, face[2] + offset])
        vertices.extend(detail_vertices)
        
        return vertices, faces

    def create_palm_assembly(self):
        """Create accurate Iron Man palm with repulsor"""
        vertices = []
        faces = []
        colors = []
        
        # Main palm - larger and more proportional
        palm_verts, palm_faces = self.create_rounded_box([0, 0, 0], [4.5, 3.5, 1.2])
        vertices.extend(palm_verts)
        faces.extend(palm_faces)
        colors.extend([[0.698, 0.133, 0.133]] * len(palm_verts))  # Deep red
        
        # Gold accent strips
        for i in range(3):
            y_pos = -1.2 + i * 1.2
            strip_verts, strip_faces = self.create_rounded_box([0, y_pos, 0.1], [4.2, 0.3, 0.2])
            offset = len(vertices)
            for face in strip_faces:
                faces.append([face[0] + offset, face[1] + offset, face[2] + offset])
            vertices.extend(strip_verts)
            colors.extend([[1.0, 0.843, 0.0]] * len(strip_verts))  # Gold
        
        # Repulsor housing in center
        repulsor_verts, repulsor_faces = self.create_sphere([0, 0, 0.4], 0.6, 12)
        offset = len(vertices)
        for face in repulsor_faces:
            faces.append([face[0] + offset, face[1] + offset, face[2] + offset])
        vertices.extend(repulsor_verts)
        colors.extend([[0.267, 0.267, 0.267]] * len(repulsor_verts))  # Gunmetal
        
        # Repulsor core
        if self.repulsor_power > 0:
            core_verts, core_faces = self.create_sphere([0, 0, 0.5], 0.4 * self.repulsor_power, 8)
            offset = len(vertices)
            for face in core_faces:
                faces.append([face[0] + offset, face[1] + offset, face[2] + offset])
            vertices.extend(core_verts)
            colors.extend([[0.0, 0.749, 1.0]] * len(core_verts))  # Blue glow
        
        # Side armor panels
        for side in [-1, 1]:
            panel_verts, panel_faces = self.create_rounded_box([side * 2.0, 0, 0], [0.8, 3.0, 1.0])
            offset = len(vertices)
            for face in panel_faces:
                faces.append([face[0] + offset, face[1] + offset, face[2] + offset])
            vertices.extend(panel_verts)
            colors.extend([[0.545, 0.0, 0.0]] * len(panel_verts))  # Dark red
        
        return vertices, faces, colors

    def create_complete_finger(self, finger_name, base_pos, joints_angles):
        """Create a complete Iron Man finger with proper segments"""
        vertices = []
        faces = []
        colors = []
        
        # Iron Man finger proportions
        if finger_name == 'thumb':
            lengths = [1.2, 1.0, 0.8]
            thickness = [0.35, 0.32, 0.28]
            base_angles = [0.6, 0, -0.3]  # Thumb positioning
        elif finger_name == 'pinky':
            lengths = [1.0, 0.9, 0.7]
            thickness = [0.28, 0.25, 0.22]
            base_angles = [0, 0, 0]
        else:
            lengths = [1.4, 1.2, 0.9]
            thickness = [0.32, 0.29, 0.25]
            base_angles = [0, 0, 0]
        
        current_pos = np.array(base_pos)
        current_direction = np.array([0, 1, 0])  # Default pointing up
        
        for i, (length, thick) in enumerate(zip(lengths, thickness)):
            # Apply joint rotation
            angle = joints_angles[i] + base_angles[i]
            
            # Calculate rotation
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
            
            current_direction = rotation_matrix @ current_direction
            end_pos = current_pos + current_direction * length
            
            # Create finger segment
            seg_verts, seg_faces = self.create_finger_segment(current_pos, end_pos, thick)
            
            # Add knuckle joint at the start
            if i < len(lengths) - 1:  # Don't add joint at fingertip
                joint_verts, joint_faces = self.create_knuckle_joint(current_pos, thick * 0.8)
                
                # Add joint
                offset = len(vertices)
                for face in joint_faces:
                    faces.append([face[0] + offset, face[1] + offset, face[2] + offset])
                vertices.extend(joint_verts)
                colors.extend([[0.267, 0.267, 0.267]] * len(joint_verts))  # Gunmetal joints
            
            # Add segment
            offset = len(vertices)
            for face in seg_faces:
                faces.append([face[0] + offset, face[1] + offset, face[2] + offset])
            vertices.extend(seg_verts)
            
            # Alternate red and gold for Iron Man look
            if i % 2 == 0:
                colors.extend([[0.698, 0.133, 0.133]] * len(seg_verts))  # Red
            else:
                colors.extend([[1.0, 0.843, 0.0]] * len(seg_verts))  # Gold
            
            current_pos = end_pos
        
        return vertices, faces, colors

    def create_wrist_assembly(self):
        """Create detailed Iron Man wrist with proper proportions"""
        vertices = []
        faces = []
        colors = []
        
        # Main wrist cylinder
        wrist_verts, wrist_faces = self.create_rounded_box([0, 0, -2.5], [2.8, 2.8, 3.0])
        vertices.extend(wrist_verts)
        faces.extend(wrist_faces)
        colors.extend([[0.545, 0.0, 0.0]] * len(wrist_verts))  # Dark red
        
        # Gold wrist bands
        for z_pos in [-3.5, -1.5]:
            band_verts, band_faces = self.create_rounded_box([0, 0, z_pos], [3.0, 3.0, 0.4])
            offset = len(vertices)
            for face in band_faces:
                faces.append([face[0] + offset, face[1] + offset, face[2] + offset])
            vertices.extend(band_verts)
            colors.extend([[1.0, 0.843, 0.0]] * len(band_verts))  # Gold
        
        # Mechanical details
        for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            x_pos = 1.6 * np.cos(angle)
            y_pos = 1.6 * np.sin(angle)
            detail_verts, detail_faces = self.create_rounded_box([x_pos, y_pos, -2.5], [0.3, 0.3, 2.5])
            offset = len(vertices)
            for face in detail_faces:
                faces.append([face[0] + offset, face[1] + offset, face[2] + offset])
            vertices.extend(detail_verts)
            colors.extend([[0.267, 0.267, 0.267]] * len(detail_verts))  # Gunmetal
        
        return vertices, faces, colors

    def set_pose(self, pose_name):
        """Set realistic Iron Man poses"""
        poses = {
            'open': {
                'thumb': [0.3, 0.1, 0.0],
                'index': [0.0, 0.0, 0.0],
                'middle': [0.0, 0.0, 0.0],
                'ring': [0.0, 0.0, 0.0],
                'pinky': [0.0, 0.0, 0.0]
            },
            'fist': {
                'thumb': [0.8, 0.9, 0.7],
                'index': [1.2, 1.4, 1.0],
                'middle': [1.2, 1.4, 1.0],
                'ring': [1.2, 1.4, 1.0],
                'pinky': [1.0, 1.2, 0.8]
            },
            'point': {
                'thumb': [0.5, 0.3, 0.1],
                'index': [0.0, 0.0, 0.0],
                'middle': [1.2, 1.4, 1.0],
                'ring': [1.2, 1.4, 1.0],
                'pinky': [1.0, 1.2, 0.8]
            },
            'repulsor': {
                'thumb': [0.6, 0.4, 0.2],
                'index': [0.3, 0.2, 0.1],
                'middle': [0.3, 0.2, 0.1],
                'ring': [0.3, 0.2, 0.1],
                'pinky': [0.4, 0.3, 0.2]
            }
        }
        
        if pose_name in poses:
            for finger, angles in poses[pose_name].items():
                self.joints[finger] = angles
            
            if pose_name == 'repulsor':
                self.repulsor_power = 0.8
            else:
                self.repulsor_power = 0.0

    def assemble_hand(self):
        """Assemble the complete Iron Man hand"""
        all_vertices = []
        all_faces = []
        all_colors = []
        
        # Create palm
        palm_verts, palm_faces, palm_colors = self.create_palm_assembly()
        all_vertices.extend(palm_verts)
        all_faces.extend(palm_faces)
        all_colors.extend(palm_colors)
        
        # Finger positions - accurate to Iron Man proportions
        finger_positions = {
            'thumb': [-1.8, -1.5, 0.2],
            'index': [1.0, 2.2, 0.1],
            'middle': [0.3, 2.3, 0.1],
            'ring': [-0.4, 2.2, 0.1],
            'pinky': [-1.1, 1.8, 0.0]
        }
        
        # Create all fingers
        for finger_name, base_pos in finger_positions.items():
            finger_verts, finger_faces, finger_colors = self.create_complete_finger(
                finger_name, base_pos, self.joints[finger_name]
            )
            
            # Offset faces for new vertices
            offset = len(all_vertices)
            for face in finger_faces:
                all_faces.append([face[0] + offset, face[1] + offset, face[2] + offset])
            
            all_vertices.extend(finger_verts)
            all_colors.extend(finger_colors)
        
        # Create wrist
        wrist_verts, wrist_faces, wrist_colors = self.create_wrist_assembly()
        offset = len(all_vertices)
        for face in wrist_faces:
            all_faces.append([face[0] + offset, face[1] + offset, face[2] + offset])
        
        all_vertices.extend(wrist_verts)
        all_colors.extend(wrist_colors)
        
        return all_vertices, all_faces, all_colors

    def animate_hand(self, frame):
        """Animate the Iron Man hand through different poses"""
        self.time = frame * 0.1
        
        # Cycle through poses
        poses = ['open', 'fist', 'point', 'repulsor', 'open']
        pose_duration = 60
        pose_index = (frame // pose_duration) % len(poses)
        self.set_pose(poses[pose_index])
        
        # Add subtle wrist movement
        self.joints['wrist_twist'] = np.sin(frame * 0.03) * 0.2
        
        # Repulsor pulse effect
        if self.pose_mode == 'repulsor':
            self.repulsor_power = 0.6 + 0.4 * np.sin(frame * 0.4)

def create_iron_man_hand():
    """Main function to create the Iron Man hand visualization"""
    hand = AccurateIronManHand()
    
    fig = plt.figure(figsize=(14, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    
    def animate(frame):
        ax.clear()
        ax.set_facecolor('black')
        
        # Update hand animation
        hand.animate_hand(frame)
        vertices, faces, colors = hand.assemble_hand()
        
        if not faces:
            return
        
        # Create 3D polygon collection
        polygons = []
        face_colors = []
        
        for i, face in enumerate(faces):
            if len(face) >= 3 and all(idx < len(vertices) for idx in face):
                face_verts = [vertices[idx] for idx in face[:3]]
                polygons.append(face_verts)
                
                # Get color for this face
                if face[0] < len(colors):
                    face_colors.append(colors[face[0]])
                else:
                    face_colors.append([0.7, 0.1, 0.1])  # Default red
        
        if polygons:
            # Apply wrist rotation
            if hand.joints['wrist_twist'] != 0:
                angle = hand.joints['wrist_twist']
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rotation_matrix = np.array([
                    [cos_a, -sin_a, 0],
                    [sin_a, cos_a, 0],
                    [0, 0, 1]
                ])
                
                # Rotate all polygons
                rotated_polygons = []
                for poly in polygons:
                    rotated_poly = [rotation_matrix @ np.array(vertex) for vertex in poly]
                    rotated_polygons.append(rotated_poly)
                polygons = rotated_polygons
            
            # Create collection
            collection = Poly3DCollection(polygons, facecolors=face_colors, 
                                        edgecolors='black', linewidths=0.3, alpha=0.9)
            ax.add_collection3d(collection)
            
            # Set proper limits
            all_points = [vertex for poly in polygons for vertex in poly]
            if all_points:
                all_points = np.array(all_points)
                center = np.mean(all_points, axis=0)
                span = np.max(all_points, axis=0) - np.min(all_points, axis=0)
                max_span = np.max(span) * 0.6
                
                ax.set_xlim([center[0] - max_span, center[0] + max_span])
                ax.set_ylim([center[1] - max_span, center[1] + max_span])
                ax.set_zlim([center[2] - max_span, center[2] + max_span])
        
        # Clean up axes
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # Remove panes
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        
        # Set viewing angle
        ax.view_init(elev=20, azim=frame * 1.5)
        
        # Add title
        pose_names = ['OPEN', 'FIST', 'POINT', 'REPULSOR', 'OPEN']
        pose_index = (frame // 60) % len(pose_names)
        ax.text2D(0.02, 0.95, f"IRON MAN MARK L - {pose_names[pose_index]} MODE", 
                 transform=ax.transAxes, color='gold', fontsize=14, fontweight='bold')
        
        if pose_names[pose_index] == 'REPULSOR':
            power_level = int(hand.repulsor_power * 100)
            ax.text2D(0.02, 0.90, f"REPULSOR POWER: {power_level}%", 
                     transform=ax.transAxes, color='cyan', fontsize=11)
    
    # Create animation
    ani = animation.FuncAnimation(fig, animate, frames=1200, interval=50, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return ani

# Run the accurate Iron Man hand
if __name__ == "__main__":
    print("Creating Movie-Accurate Iron Man Hand...")
    print("Features:")
    print("✓ Accurate proportions and design")
    print("✓ Proper red and gold color scheme")
    print("✓ Realistic finger articulation")
    print("✓ Multiple authentic poses")
    print("✓ Animated repulsor effects")
    print("✓ Mechanical joint details")
    print("✓ 360° rotating camera view")
    
    animation = create_iron_man_hand()