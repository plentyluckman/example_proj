import numpy as np
import os

from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Kinematic chain using indices to match parent array
# Parent array: [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]

joint_indices = {i: i for i in range(22)}

# Measured bone lengths
BONE_LENGTHS_VIBE = {
    (0, 1): 0.1215,  # Root to left hip
    (1, 2): 0.4821,  # Left hip to left knee
    (2, 3): 0.4339,  # Left knee to left ankle
    (3, 4): 0.2416,  # Left ankle to foot
    (0, 5): 0.1205,  # Root to right hip
    (5, 6): 0.4847,  # Right hip to right knee
    (6, 7): 0.4458,  # Right knee to right ankle
    (7, 8): 0.2431,  # Right ankle to foot
    (0, 9): 0.1256,  # Root to spine start
    (9, 10): 0.1256, # Lower spine
    (10, 11): 0.0915, # Mid spine
    (11, 12): 0.1483, # Upper spine to neck
    (12, 13): 0.2080, # Neck to head
    (11, 14): 0.1112, # Spine to left collar
    (14, 15): 0.0821, # Left collar to shoulder
    (15, 16): 0.2440, # Left upper arm
    (16, 17): 0.2449, # Left lower arm
    (11, 18): 0.1159, # Spine to right collar
    (18, 19): 0.0801, # Right collar to shoulder
    (19, 20): 0.2528, # Right upper arm
    (20, 21): 0.2375  # Right lower arm
}


BONE_DIRECTIONS_VIBE = {
    # Spine chain - generally goes up
    (0, 9): np.array([0., 0., 1.]),     # Root to spine1
    (9, 10): np.array([0., 0., 1.]),    # Spine1 to spine2
    (10, 11): np.array([0., 0., 1.]),   # Spine2 to spine3
    (11, 12): np.array([0., 0., 1.]),   # Spine3 to neck
    (12, 13): np.array([0., 0., 1.]),   # Neck to head
    
    # Left side - updated hip direction
    (0, 1): np.array([-1.0, 0., -0.1]),  # Root to left hip - more horizontal
    (1, 2): np.array([0., 0., -1.]),     # Left hip to knee
    (2, 3): np.array([0., 0., -1.]),     # Left knee to ankle
    (3, 4): np.array([0., 0.3, -0.1]),   # Left ankle to foot
    
    # Right side - updated hip direction
    (0, 5): np.array([1.0, 0., -0.1]),   # Root to right hip - more horizontal
    (5, 6): np.array([0., 0., -1.]),     # Right hip to knee
    (6, 7): np.array([0., 0., -1.]),     # Right knee to ankle
    (7, 8): np.array([0., 0.3, -0.1]),   # Right ankle to foot
    
    # Left arm chain
    (11, 14): np.array([-0.8, 0., 0.2]), # Spine to left collar
    (14, 15): np.array([-1., 0., 0.]),   # Left collar to shoulder
    (15, 16): np.array([-1., 0., -0.1]), # Left shoulder to elbow
    (16, 17): np.array([-1., 0., -0.2]), # Left elbow to wrist
    
    # Right arm chain
    (11, 18): np.array([0.8, 0., 0.2]),  # Spine to right collar
    (18, 19): np.array([1., 0., 0.]),    # Right collar to shoulder
    (19, 20): np.array([1., 0., -0.1]),  # Right shoulder to elbow
    (20, 21): np.array([1., 0., -0.2]),  # Right elbow to wrist
}

# LaFAN1 bone lengths derived from dataset analysis
BONE_LENGTHS_LAFAN1 = {
    # Root connections - using conservative averages from data
    (0, 1): 0.1000,  # Root -> L_Hip (shortened to reduce instability)
    (0, 5): 0.1000,  # Root -> R_Hip
    (0, 9): 0.1000,  # Root -> Spine1
    
    # Left leg chain - consistent values from data
    (1, 2): 0.4470,  # L_Hip -> L_Knee
    (2, 3): 0.4430,  # L_Knee -> L_Ankle (adjusted from anomalous 1.1278)
    (3, 4): 0.2507,  # L_Ankle -> L_Foot
    
    # Right leg chain - symmetric with left
    (5, 6): 0.4470,  # R_Hip -> R_Knee
    (6, 7): 0.4430,  # R_Knee -> R_Ankle
    (7, 8): 0.2507,  # R_Ankle -> R_Foot
    
    # Spine chain
    (9, 10): 0.0625,  # Spine1 -> Spine2
    (10, 11): 0.0245,  # Spine2 -> Spine3 (adjusted from 0.2449)
    (11, 12): 0.1349,  # Spine3 -> Neck
    (12, 13): 0.1407,  # Neck -> Head
    
    # Left arm chain
    (11, 14): 0.0964,  # Spine3 -> L_Collar
    (14, 15): 0.1048,  # L_Collar -> L_Shoulder
    (15, 16): 0.2172,  # L_Shoulder -> L_Elbow
    (16, 17): 0.0780,  # L_Elbow -> L_Wrist
    
    # Right arm chain 
    (11, 18): 0.0964,  # Spine3 -> R_Collar
    (18, 19): 0.1048,  # R_Collar -> R_Shoulder
    (19, 20): 0.2172,  # R_Shoulder -> R_Elbow
    (20, 21): 0.0780   # R_Elbow -> R_Wrist
}

# Canonical bone directions for LaFAN1
BONE_DIRECTIONS_LAFAN1 = {
    # Spine chain - generally goes up
    (0, 9): np.array([0., 0., 1.]),     # Root to spine1
    (9, 10): np.array([0., 0., 1.]),    # Spine1 to spine2
    (10, 11): np.array([0., 0., 1.]),   # Spine2 to spine3
    (11, 12): np.array([0., 0., 1.]),   # Spine3 to neck
    (12, 13): np.array([0., 0., 1.]),   # Neck to head
    
    # Left leg chain
    (0, 1): np.array([-0.2, 0., -0.1]),  # Root to left hip
    (1, 2): np.array([0., 0., -1.]),     # Left hip to knee
    (2, 3): np.array([0., 0., -1.]),     # Left knee to ankle
    (3, 4): np.array([0.1, 0.3, -0.1]),  # Left ankle to foot
    
    # Right leg chain
    (0, 5): np.array([0.2, 0., -0.1]),   # Root to right hip
    (5, 6): np.array([0., 0., -1.]),     # Right hip to knee
    (6, 7): np.array([0., 0., -1.]),     # Right knee to ankle
    (7, 8): np.array([-0.1, 0.3, -0.1]), # Right ankle to foot
    
    # Left arm chain
    (11, 14): np.array([-0.7, 0., 0.1]), # Spine to left collar
    (14, 15): np.array([-1., 0., 0.]),   # Left collar to shoulder
    (15, 16): np.array([-1., 0., -0.1]), # Left shoulder to elbow
    (16, 17): np.array([-1., 0., -0.2]), # Left elbow to wrist
    
    # Right arm chain
    (11, 18): np.array([0.7, 0., 0.1]),  # Spine to right collar
    (18, 19): np.array([1., 0., 0.]),    # Right collar to shoulder
    (19, 20): np.array([1., 0., -0.1]),  # Right shoulder to elbow
    (20, 21): np.array([1., 0., -0.2]),  # Right elbow to wrist
}

def get_default_bone_direction(parent_idx, child_idx, bone_directions=BONE_DIRECTIONS_VIBE):
    """Get normalized bone direction"""
    direction = bone_directions.get((parent_idx, child_idx), np.array([0., 0., 1.]))
    return direction / np.linalg.norm(direction)

def quaternion_fk(joint_angles, parents, root_position=None, bone_lengths=BONE_LENGTHS_VIBE, bone_directions=BONE_DIRECTIONS_VIBE):
    """
    Forward kinematics using quaternions with global root position
    Args:
        joint_angles: array of shape (num_joints * 3) containing euler angles
        parents: array of parent indices
        root_position: Optional global position for root joint
    Returns:
        positions: array of shape (num_joints, 3) containing joint positions
    """
    num_joints = len(parents)
    positions = np.zeros((num_joints, 3))
    orientations = []
    
    # Set root position if provided
    if root_position is not None:
        positions[0] = root_position
    
    # Initialize root
    root_rot = R.from_euler('xyz', joint_angles[:3])
    orientations.append(root_rot)
    
    # Process other joints
    for i in range(1, num_joints):
        parent = parents[i]
        
        # Get current joint rotation
        euler_angles = joint_angles[i*3:(i+1)*3]
        local_rot = R.from_euler('xyz', euler_angles)
        
        # Compute global orientation
        if parent != -1:
            global_rot = orientations[parent] * local_rot
        else:
            global_rot = local_rot
            
        orientations.append(global_rot)
        
        # Compute position
        if parent != -1:
            bone_length = bone_lengths.get((parent, i), 0.1)
            direction = get_default_bone_direction(parent, i, bone_directions)
            
            # Scale direction by bone length
            offset = direction * bone_length
            
            # Apply parent rotation to offset
            rotated_offset = orientations[parent].apply(offset)
            positions[i] = positions[parent] + rotated_offset
    
    return positions

def get_target_quaternion_magnitudes():
    """Define target quaternion vector magnitude ranges for each joint"""
    return {
        # Root and spine - significant rotations 
        0: (0.4, 0.8),   # Root 
        9: (0.3, 0.6),   # Spine1
        10: (0.3, 0.6),  # Spine2
        11: (0.3, 0.6),  # Spine3
        12: (0.4, 0.8),  # Neck
        
        # Arms - very large rotations
        14: (0.5, 0.9),  # Left shoulder
        15: (0.5, 0.9),  # Left elbow  
        16: (0.4, 0.8),  # Left wrist
        18: (0.5, 0.9),  # Right shoulder
        19: (0.5, 0.9),  # Right elbow
        20: (0.4, 0.8),  # Right wrist
        
        # Legs - large rotations 
        1: (0.5, 0.9),   # Left hip
        2: (0.5, 0.9),   # Left knee
        3: (0.4, 0.8),   # Left ankle
        5: (0.5, 0.9),   # Right hip  
        6: (0.5, 0.9),   # Right knee
        7: (0.4, 0.8),   # Right ankle
        
        # End effectors - medium rotations
        4: (0.2, 0.5),   # Left foot
        8: (0.2, 0.5),   # Right foot  
        13: (0.2, 0.5),  # Head
        17: (0.2, 0.5),  # Left hand
        21: (0.2, 0.5)   # Right hand
    }

def quaternion_magnitude_error(euler_angles, target_range):
    """Compute error based on quaternion vector magnitude"""
    min_mag, max_mag = target_range
    
    # Convert euler to quaternion
    quat = euler_to_quat(euler_angles[None])[0]
    
    # Get magnitude of vector part (x,y,z)
    vec_magnitude = np.linalg.norm(quat[1:])
    
    if vec_magnitude < min_mag:
        return (min_mag - vec_magnitude) * 20.0  # Stronger penalty
    elif vec_magnitude > max_mag:
        return (vec_magnitude - max_mag) * 10.0
    return 0.0

def objective_function(joint_angles, target_positions, parents, prev_angles=None):
    """Modified objective function using quaternion magnitudes"""
    num_joints = len(parents)
    root_pos = target_positions[0]
    current_positions = quaternion_fk(joint_angles, parents, root_position=root_pos)
    
    # Base position error
    position_errors = (current_positions - target_positions)**2
    position_error = np.sum(position_errors)
    
    # Quaternion magnitude constraints
    quat_error = 0.0
    constraints = get_target_quaternion_magnitudes()
    
    for joint_idx in range(num_joints):
        euler = joint_angles[joint_idx*3:(joint_idx+1)*3]
        target_range = constraints[joint_idx]
        quat_error += quaternion_magnitude_error(euler, target_range)
    
    # Temporal smoothing
    temporal_error = 0.0
    if prev_angles is not None:
        # Only apply temporal smoothing to non-key rotations
        temporal_diff = joint_angles - prev_angles
        temporal_error = np.sum(temporal_diff**2)
    
    # Weight quaternion error more heavily than position
    total_error = position_error + quat_error * 10.0 + temporal_error * 0.5
    return total_error

def inverse_kinematics(target_positions, parents, initial_guess=None, prev_angles=None):
    """Enhanced IK focusing on quaternion magnitudes"""
    num_joints = len(parents)
    constraints = get_target_quaternion_magnitudes()
    
    if initial_guess is None:
        # Initialize with random rotations within target ranges
        initial_guess = np.zeros(num_joints * 3)
        for joint_idx in range(num_joints):
            min_mag, max_mag = constraints[joint_idx]
            
            # Generate random angles that will produce quaternions
            # with vector magnitudes in the target range
            angle = np.arccos(1 - min_mag**2) * 2
            
            # Distribute rotation across axes
            axis = np.random.normal(0, 1, 3)
            axis = axis / np.linalg.norm(axis) * angle
            initial_guess[joint_idx*3:joint_idx*3+3] = axis
    
    # Multiple optimization passes
    best_result = None
    best_error = float('inf')
    
    for i in range(5):  # More optimization passes
        result = minimize(
            objective_function,
            initial_guess,
            args=(target_positions, parents, prev_angles),
            method='L-BFGS-B',
            bounds=[(-np.pi, np.pi)] * (num_joints * 3),
            options={'maxiter': 500, 'ftol': 1e-8}
        )
        
        if result.fun < best_error:
            best_error = result.fun
            best_result = result.x
        
        # Generate new initial guess
        if i < 4:  # Skip last iteration
            initial_guess = best_result.copy()
            # Add random perturbations
            noise = np.random.normal(0, 0.2, best_result.shape)
            initial_guess += noise
    
    return best_result

def debug_quaternions(joint_angles, constraints=None):
    """Debug helper to analyze quaternion magnitudes"""
    if constraints is None:
        constraints = get_target_quaternion_magnitudes()
        
    print("\nQuaternion Analysis:")
    print("Joint |  Vec Mag  | Target Range | Status")
    print("-" * 45)
    
    for joint_idx in range(len(joint_angles) // 3):
        euler = joint_angles[joint_idx*3:(joint_idx+1)*3]
        quat = euler_to_quat(euler[None])[0]
        vec_mag = np.linalg.norm(quat[1:])
        
        min_mag, max_mag = constraints[joint_idx]
        
        if vec_mag < min_mag:
            status = "TOO SMALL"
        elif vec_mag > max_mag:
            status = "TOO LARGE"
        else:
            status = "OK"
            
        print(f"{joint_idx:5d} | {vec_mag:8.3f} | {min_mag:4.2f}-{max_mag:4.2f} | {status}")
    
    return vec_mag  # Return last magnitude for testing

def process_motion_sequence(positions, parents):
    """Process sequence with quaternion-focused optimization"""
    num_frames = positions.shape[0]
    num_joints = positions.shape[1]
    joint_angles_all = np.zeros((num_frames, num_joints * 3))
    
    for i in range(num_frames):
        if i % 10 == 0:
            print(f"\nProcessing frame {i}/{num_frames}")
        
        prev_angles = joint_angles_all[i-1] if i > 0 else None
        joint_angles = inverse_kinematics(
            positions[i], 
            parents,
            initial_guess=prev_angles,
            prev_angles=prev_angles
        )
        joint_angles_all[i] = joint_angles
        
        # Debug every 10 frames
        if i % 10 == 0:
            debug_quaternions(joint_angles)
    
    return joint_angles_all

def animate_skeleton(joint_angles, positions, parents, interval=50, save_file=None, bone_lengths=BONE_LENGTHS_VIBE , bone_directions=BONE_DIRECTIONS_VIBE):
    """
    Create an animation of the skeleton movement with global motion
    Args:
        joint_angles: array of shape (num_frames, num_joints * 3)
        positions: original global positions array
        parents: array of parent indices
        interval: time between frames in milliseconds
        save_file: if provided, save the animation to this file
    """
    num_frames = joint_angles.shape[0]
    num_joints = len(parents)
    
    # Joint names for labels
    joint_names = {
        0: "Root", 1: "L_Hip", 2: "L_Knee", 3: "L_Ankle", 4: "L_Foot",
        5: "R_Hip", 6: "R_Knee", 7: "R_Ankle", 8: "R_Foot",
        9: "Spine1", 10: "Spine2", 11: "Spine3", 12: "Neck", 13: "Head",
        14: "L_Collar", 15: "L_Shoulder", 16: "L_Elbow", 17: "L_Wrist",
        18: "R_Collar", 19: "R_Shoulder", 20: "R_Elbow", 21: "R_Wrist"
    }
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame):
        ax.clear()
        
        # Get current positions using FK with global root position
        current_positions = quaternion_fk(
            joint_angles[frame], 
            parents, 
            root_position=positions[frame, 0],
            bone_lengths = bone_lengths,
            bone_directions= bone_directions
        )
        
        # Draw bones with colors
        for i in range(1, num_joints):
            if parents[i] != -1:
                start = current_positions[parents[i]]
                end = current_positions[i]
                
                # Color based on body part
                if i in [1, 2, 3, 4, 14, 15, 16, 17]:  # Left side
                    color = 'blue'
                elif i in [5, 6, 7, 8, 18, 19, 20, 21]:  # Right side
                    color = 'red'
                else:  # Spine and head
                    color = 'green'
                
                ax.plot([start[0], end[0]], 
                       [start[1], end[1]], 
                       [start[2], end[2]], 
                       color=color, linewidth=2)
        
        # Plot joint points
        ax.scatter(current_positions[:, 0], 
                  current_positions[:, 1], 
                  current_positions[:, 2], 
                  c='black', s=20)
        
        # Add joint labels
        for i, pos in enumerate(current_positions):
            ax.text(pos[0], pos[1], pos[2], 
                   joint_names[i], fontsize=8)
            """Get normalized bone direction"""
        # Set plot limits with margins
        margin = 0.5
        ax.set_xlim(current_positions[:, 0].min() - margin, 
                   current_positions[:, 0].max() + margin)
        ax.set_ylim(current_positions[:, 1].min() - margin, 
                   current_positions[:, 1].max() + margin)
        ax.set_zlim(current_positions[:, 2].min() - margin, 
                   current_positions[:, 2].max() + margin)
        
        # Set view angle
        ax.view_init(elev=20, azim=45)
        ax.set_title(f'Frame {frame}')
        
        return ax.get_children()
    
    anim = FuncAnimation(
        fig, update, frames=num_frames, 
        interval=interval, blit=True
    )
    
    if save_file:
        anim.save('ik_recon_001.gif', writer='pillow', fps=30)
    
    plt.show()


def animate_skeleton_globalpos(positions, parents, interval=50, save_file=None):

    num_frames = positions.shape[0]
    num_joints = len(parents)
    
    # Joint names for labels
    joint_names = {
        0: "Root", 1: "L_Hip", 2: "L_Knee", 3: "L_Ankle", 4: "L_Foot",
        5: "R_Hip", 6: "R_Knee", 7: "R_Ankle", 8: "R_Foot",
        9: "Spine1", 10: "Spine2", 11: "Spine3", 12: "Neck", 13: "Head",
        14: "L_Collar", 15: "L_Shoulder", 16: "L_Elbow", 17: "L_Wrist",
        18: "R_Collar", 19: "R_Shoulder", 20: "R_Elbow", 21: "R_Wrist"
    }
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame):
        ax.clear()
        
        # Get current positions using FK with global root position
        current_positions = positions[frame]
        
        # Draw bones with colors
        for i in range(1, num_joints):
            if parents[i] != -1:
                start = current_positions[parents[i]]
                end = current_positions[i]
                
                # Color based on body part
                if i in [1, 2, 3, 4, 14, 15, 16, 17]:  # Left side
                    color = 'blue'
                elif i in [5, 6, 7, 8, 18, 19, 20, 21]:  # Right side
                    color = 'red'
                else:  # Spine and head
                    color = 'green'
                
                ax.plot([start[0], end[0]], 
                       [start[1], end[1]], 
                       [start[2], end[2]], 
                       color=color, linewidth=2)
        
        # Plot joint points
        ax.scatter(current_positions[:, 0], 
                  current_positions[:, 1], 
                  current_positions[:, 2], 
                  c='black', s=20)
        
        # Add joint labels
        for i, pos in enumerate(current_positions):
            ax.text(pos[0], pos[1], pos[2], 
                   joint_names[i], fontsize=8)
            """Get normalized bone direction"""
        # Set plot limits with margins
        margin = 0.5
        ax.set_xlim(current_positions[:, 0].min() - margin, 
                   current_positions[:, 0].max() + margin)
        ax.set_ylim(current_positions[:, 1].min() - margin, 
                   current_positions[:, 1].max() + margin)
        ax.set_zlim(current_positions[:, 2].min() - margin, 
                   current_positions[:, 2].max() + margin)
        
        # Set view angle
        ax.view_init(elev=20, azim=45)
        ax.set_title(f'Frame {frame}')
        
        return ax.get_children()
    
    anim = FuncAnimation(
        fig, update, frames=num_frames, 
        interval=interval, blit=True
    )
    
    if save_file:
        anim.save('original_blanket.gif', writer='pillow', fps=30)
    
    plt.show()

def process_lafan1_motion_sequence(local_rot):
    """Convert LaFAN1 local quaternion rotations to Euler angles.
    
    Args:
        local_rot: Local quaternion rotations with shape (T, J, 4) in (w,x,y,z) format
        
    Returns:
        joint_angles: Euler angles with shape (T, J*3) in xyz format
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    
    num_frames, num_joints, _ = local_rot.shape
    joint_angles = np.zeros((num_frames, num_joints * 3))
    
    # Convert quaternions from (w,x,y,z) to (x,y,z,w) format for scipy
    scipy_quats = np.zeros_like(local_rot)
    scipy_quats[..., [0,1,2,3]] = local_rot[..., [1,2,3,0]]  # Reorder components
    
    # Process each frame
    for t in range(num_frames):
        # Process each joint
        for j in range(num_joints):
            # Create rotation object from quaternion
            rot = R.from_quat(scipy_quats[t, j])
            
            # Convert to xyz Euler angles (in radians)
            euler = rot.as_euler('xyz', degrees=False)
            
            # Store in flattened array
            joint_angles[t, j*3:(j+1)*3] = euler
            
    return joint_angles


def visualize_motion(batch_file, save_path="visualization/motion.mp4", fps=30):
    """
    Visualize motion sequence as an animation with proper orientation
    
    Args:
        batch_file: Path to numpy batch file
        save_path: Path to save the animation (mp4)
        fps: Frames per second for the animation
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Load batch data
    batch = np.load(batch_file, allow_pickle=True).item()
    positions = batch['positions'][0]  # Local positions
    rotations = batch['rotations'][0]  # Local rotations
    parents = batch['parents']  # Joint hierarchy

    # Apply forward kinematics to get global positions
    global_rotations, global_positions = quat_fk(rotations, positions, parents)
    positions = global_positions  # Use global positions for visualization
    
    # Rotate coordinates to make Z up
    # Assuming input is Y-up, we want to rotate -90 degrees around X axis
    positions_rotated = positions.copy()
    positions_rotated[..., 1], positions_rotated[..., 2] = -positions[..., 2], positions[..., 1]

    # Define connections based on the parent-child relationships
    connections = []
    for i, parent in enumerate(parents):
        if parent != -1:  # Skip root
            connections.append((parent, i))

    # Create mask for past and future frames
    n_frames = positions.shape[0]
    n_past = 10  # Number of past frames
    n_future = 10  # Number of future frames
    mask = np.zeros(n_frames)
    mask[:n_past] = 1  # Past context
    mask[-n_future:] = 1  # Future context

    # Setup figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get axis limits
    margin = 0.5
    min_val = np.min(positions_rotated)
    max_val = np.max(positions_rotated)
    range_val = max_val - min_val
    mid_val = (max_val + min_val) / 2
    
    def update(frame):
        ax.clear()
        
        # Set consistent axis limits
        ax.set_xlim(mid_val - range_val, mid_val + range_val)
        ax.set_ylim(mid_val - range_val, mid_val + range_val)
        ax.set_zlim(min_val - margin, max_val + margin)
        
        # Get positions for current frame
        pos = positions_rotated[frame]
        
        # Plot joints
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='black', s=20)
        
        # Plot bones
        is_keyframe = bool(mask[frame])
        color = 'blue' if is_keyframe else 'red'
        linewidth = 3 if is_keyframe else 2
        
        for start_idx, end_idx in connections:
            start_pos = pos[start_idx]
            end_pos = pos[end_idx]
            ax.plot([start_pos[0], end_pos[0]],
                   [start_pos[1], end_pos[1]],
                   [start_pos[2], end_pos[2]],
                   c=color, linewidth=linewidth)
        
        # Set labels and adjust view
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {frame} - {"Keyframe" if is_keyframe else "Interpolated"}')
        ax.view_init(elev=15, azim=45)
        ax.grid(False)
        
        return ax

    # Create animation
    anim = FuncAnimation(fig, update, frames=n_frames, 
                                 interval=1000/fps, blit=False)
    
    # Save animation
    anim.save(save_path, writer='ffmpeg', fps=fps)
    plt.close()

############################################################################
###########LATER SHOULD BE REMOVED and imported from the module#############
############################################################################

def quat_fk(lrot, lpos, parents):
    """
    Performs Forward Kinematics (FK) on local quaternions and local positions to retrieve global representations

    :param lrot: tensor of local quaternions with shape (..., Nb of joints, 4)
    :param lpos: tensor of local positions with shape (..., Nb of joints, 3)
    :param parents: list of parents indices
    :return: tuple of tensors of global quaternion, global positions
    """
    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(quat_mul_vec(gr[parents[i]], lpos[..., i:i+1, :]) + gp[parents[i]])
        gr.append(quat_mul    (gr[parents[i]], lrot[..., i:i+1, :]))

    res = np.concatenate(gr, axis=-2), np.concatenate(gp, axis=-2)
    return res


def quat_mul(x, y):
    """
    Performs quaternion multiplication on arrays of quaternions

    :param x: tensor of quaternions of shape (..., Nb of joints, 4)
    :param y: tensor of quaternions of shape (..., Nb of joints, 4)
    :return: The resulting quaternions
    """
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    res = np.concatenate([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], axis=-1)

    return res


def quat_mul_vec(q, x):
    """
    Performs multiplication of an array of 3D vectors by an array of quaternions (rotation).

    :param q: tensor of quaternions of shape (..., Nb of joints, 4)
    :param x: tensor of vectors of shape (..., Nb of joints, 3)
    :return: the resulting array of rotated vectors
    """
    t = 2.0 * np.cross(q[..., 1:], x)
    res = x + q[..., 0][..., np.newaxis] * t + np.cross(q[..., 1:], t)

    return res


def angle_axis_to_quat(angle, axis):
    """
    Converts from and angle-axis representation to a quaternion representation

    :param angle: angles tensor
    :param axis: axis tensor
    :return: quaternion tensor
    """
    c = np.cos(angle / 2.0)[..., np.newaxis]
    s = np.sin(angle / 2.0)[..., np.newaxis]
    q = np.concatenate([c, s * axis], axis=-1)
    return q


def euler_to_quat(e, order='zyx'):
    """

    Converts from an euler representation to a quaternion representation

    :param e: euler tensor
    :param order: order of euler rotations
    :return: quaternion tensor
    """
    axis = {
        'x': np.asarray([1, 0, 0], dtype=np.float32),
        'y': np.asarray([0, 1, 0], dtype=np.float32),
        'z': np.asarray([0, 0, 1], dtype=np.float32)}

    q0 = angle_axis_to_quat(e[..., 0], axis[order[0]])
    q1 = angle_axis_to_quat(e[..., 1], axis[order[1]])
    q2 = angle_axis_to_quat(e[..., 2], axis[order[2]])

    return quat_mul(q0, quat_mul(q1, q2))

def remove_quat_discontinuities(rotations):
    """

    Removing quat discontinuities on the time dimension (removing flips)

    :param rotations: Array of quaternions of shape (T, J, 4)
    :return: The processed array without quaternion inversion.
    """
    rots_inv = -rotations

    for i in range(1, rotations.shape[0]):
        # Compare dot products
        replace_mask = np.sum(rotations[i - 1: i] * rotations[i: i + 1], axis=-1) < np.sum(
            rotations[i - 1: i] * rots_inv[i: i + 1], axis=-1)
        replace_mask = replace_mask[..., np.newaxis]
        rotations[i] = replace_mask * rots_inv[i] + (1.0 - replace_mask) * rotations[i]

    return rotations


############################################################################

def compute_local_positions_vibe(global_positions, parents):
    """Convert VIBE global positions to local positions"""
    num_frames, num_joints, _ = global_positions.shape
    local_positions = np.zeros_like(global_positions)
    
    # Root stays the same (global)
    local_positions[:, 0] = global_positions[:, 0]
    
    # Other joints relative to parent
    for j in range(1, num_joints):
        parent = parents[j]
        if parent != -1:
            local_positions[:, j] = global_positions[:, j] - global_positions[:, parent]
            
    return local_positions


def process_vibe_to_lafan_format(global_positions, joint_angles, parents):
    """Convert VIBE data to LaFAN1 format with correct scaling and rotations"""
    
    # 1. Fix rotation conversion
    def convert_euler_to_quat_like_lafan(euler_angles, order='xyz'):
        """Convert Euler angles to quaternions using LaFAN1's approach"""
        num_frames = euler_angles.shape[0]
        num_joints = euler_angles.shape[1] // 3
        
        # Reshape euler angles
        euler_reshaped = np.radians(euler_angles.reshape(num_frames, num_joints, 3))
        
        # Convert using their functions 
        print(f"joint_angles {euler_reshaped[10,2:4,:]}")
        quats = euler_to_quat(euler_reshaped, order=order)
        print(f"joint_angles2 {quats[10,2:4,:]}")
        quats = remove_quat_discontinuities(quats)
        print(f"joint_angles3 {quats[10,2:4,:]}")

        
        return quats
        
    # 2. Fix local positions with correct scaling
    def compute_local_positions_scaled(global_pos, parents):
        local_pos = np.zeros_like(global_pos)
        local_pos[:, 0] = global_pos[:, 0]  # Root stays global
        
        for j in range(1, global_pos.shape[1]):
            parent = parents[j]
            if parent != -1:
                # Get local offset from parent
                local_pos[:, j] = global_pos[:, j] - global_pos[:, parent]
                
        # Scale to match LaFAN1 range
        scale_factor = 100  # Adjust based on your data
        local_pos = local_pos * scale_factor
        
        return local_pos
    
    print("Converting VIBE format to LaFAN1...")
    
    # Convert rotations
    local_rot_vibe = convert_euler_to_quat_like_lafan(joint_angles)
    
    
    # Convert and scale positions
    local_pos_vibe = compute_local_positions_scaled(global_positions, parents)
    
    # Ensure everything is numpy arrays
    vibe_data = {
        'positions': np.array(local_pos_vibe[None]),  # Add batch dim
        'rotations': np.array(local_rot_vibe[None]),  # Add batch dim
        'parents': np.array(parents)
    }
    
    return vibe_data


def debug_comparison(vibe_data, sample_lafan_data):
    print("\nComparing VIBE vs LaFAN1 format:")
    
    # Compare rotations first
    print("\nRotations comparison (first frame, first few joints):")
    print("LaFAN1 quats:")
    print(sample_lafan_data['rotations'][0, 10, :3])  # First 3 joints
    print("\nVIBE converted quats:")
    print(vibe_data['rotations'][0, 10, :3])  # First 3 joints
    
    # Compare positions
    print("\nPositions comparison (first frame, first few joints):")
    print("LaFAN1 local positions:")
    print(sample_lafan_data['positions'][0, 10, :3])
    print("\nVIBE local positions:")
    print(vibe_data['positions'][0, 10, :3])
    
    # Compare global positions after FK
    lafan_global_rot, lafan_global_pos = quat_fk(
        sample_lafan_data['rotations'][0], 
        sample_lafan_data['positions'][0], 
        sample_lafan_data['parents']
    )
    
    vibe_global_rot, vibe_global_pos = quat_fk(
        vibe_data['rotations'][0],
        vibe_data['positions'][0],
        vibe_data['parents']
    )
    
    print("\nGlobal positions after FK:")
    print("LaFAN1:", lafan_global_pos[0, :3])
    print("VIBE:", vibe_global_pos[0, :3])
    
    return lafan_global_pos, vibe_global_pos


if __name__ == '__main__':
    from basic_vibe_preprocessing import load_vibe_data, process_vibe_joints, convert_to_lafan1_style, PARENTS
    print("ROOT PATH == ", os.getcwd())
    PATH = './data/blanket_occlusion.pkl'
    data0 = load_vibe_data(PATH, 0)
    motion_data, _, _ = data0
    # Extract LaFAN1-style motion dataset
    processed_joints = process_vibe_joints(motion_data) 
    lafan1_motion, parents = convert_to_lafan1_style(processed_joints)    
    print(f"lafan1_motion_all SHAPE: {lafan1_motion.shape}")
    
    # After processing the VIBE data and getting joint_angles:
    MODE = ['draw_original', 'draw_lafan1', 'vibe_ik', 'convert_vibe'] #several options to select
    # mode_idx = 1
    mode_idx =3
    mode_main = MODE[mode_idx]
    if mode_main == 'draw_original':
        animate_skeleton_globalpos(lafan1_motion, parents, interval=50, save_file=True)
    elif mode_main == 'draw_lafan1':
        import torch
        #Lafan1 comparison
        test_dir= os.path.join(os.getcwd(), 'dataset','lafan1', 'numpy','test')
        assert os.path.exists(test_dir)
        test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.npy')])
        # choose 2000th batch (just random) for our jorney of testing..
        choose = test_files[2000]
        batch_file = os.path.join(test_dir, choose)
        batch = np.load(batch_file, allow_pickle=True).item()
        batch_data = {k: torch.tensor(v).float() if isinstance(v, np.ndarray) else v 
                        for k, v in batch.items()}

        lpos = batch_data['positions'].squeeze()
        lrot = batch_data['rotations'].squeeze()
        assert lpos.size() == (65, 22,3) and lrot.size() == (65, 22,4)
        # Convert rotations
        joint_angles_all = process_lafan1_motion_sequence(lrot)
        assert joint_angles_all.shape == (lrot.shape[0], 66)

        # Use joint angles for animation
        # animate_skeleton(joint_angles_all, lpos, parents, interval=50, save_file=True, bone_lengths=BONE_LENGTHS_LAFAN1, bone_directions=BONE_DIRECTIONS_LAFAN1) #not works..
        visualize_motion(batch_file, './motion.gif')

        vibe_data = np.load('./vibe_blanket.npy', allow_pickle=True).item()
        debug_comparison(vibe_data, batch_data)
        
    elif mode_main == 'vibe_ik':
        joint_angles_all = process_motion_sequence(lafan1_motion, parents)
        animate_skeleton(joint_angles_all, lafan1_motion, parents, interval=50, save_file=True)
        print(f"joint_angles_all SHAPE: {joint_angles_all.shape}")
    elif mode_main == 'convert_vibe':
        # Convert to local representation
        run_ik = True
        EXAMPLE = './vibe_blanket.npy'
        if run_ik:
            WINDOW_SIZE = 65
            FROM = 200
            UPTO = FROM + WINDOW_SIZE
            sliced_data = lafan1_motion[FROM:UPTO, ...]
            joint_angles_all = process_motion_sequence(sliced_data, parents)
            vibe_data = process_vibe_to_lafan_format(sliced_data, joint_angles_all, parents)
            np.save(EXAMPLE, vibe_data)

        # Test visualization
        visualize_motion(EXAMPLE, './vibe_motion_blanket.gif')


        
    # np.save(joint_angles_all, 'rot_euler_blanket.npy')