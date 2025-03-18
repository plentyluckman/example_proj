import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

class BVHSkeleton:
    def __init__(self, s: str):
        self.num_joints = 0
        self.joint_names = []
        self.parent_idx = []
        self.offsets = []
        self.rot_ch_order = ''
        self.num_ch = 0
        self.channels = []
        
        self.definition_str = s

        self.parse(s)

    def parse(self, s: str):
        cur_joint_stack = []
        
        lines = s.split('\n')

        for line in lines:
            line = line.strip()
            if 'HIERARCHY' in line:
                continue
            if 'ROOT' in line or 'JOINT' in line or 'End Site' in line:
                if 'End Site' in line:
                    self.joint_names.append(self.joint_names[cur_joint_stack[-1]] + '_EndEffector')
                    self.channels.append(0)
                else:
                    self.joint_names.append(line.split(' ')[1])
                self.parent_idx.append(-1 if len(cur_joint_stack) == 0 else cur_joint_stack[-1])
                self.offsets.append(np.zeros(3))
                cur_joint_stack.append(self.num_joints)
                self.num_joints += 1

            if 'OFFSET' in line:
                self.offsets[cur_joint_stack[-1]] = np.array(list(map(float, line.split(' ')[1:])))

            if 'CHANNELS' in line:
                channel_num = int(line.split(' ')[1])
                self.channels.append(channel_num)
                if len(self.rot_ch_order) == 0:
                    # First time
                    if channel_num == 6:
                        channel_strings = line.split(' ')[5:]
                    elif channel_num == 3:
                        channel_strings = line.split(' ')[2:]
                    else:
                        raise NotImplementedError('Only 3 or 6 channels are supported')
                    self.rot_ch_order = ''.join(list(map(lambda x: x[0].upper(), channel_strings)))
                else:
                    if channel_num == 6:
                        channel_strings = line.split(' ')[5:]
                    elif channel_num == 3:
                        channel_strings = line.split(' ')[2:]
                    else:
                        raise NotImplementedError('Only 3 or 6 channels are supported')
                    
                    if ''.join(list(map(lambda x: x[0].upper(), channel_strings))) != self.rot_ch_order:
                        raise NotImplementedError('Different channels are not supported')
            if '{' in line:
                continue

            if '}' in line:
                cur_joint_stack.pop()
                continue
        
        self.num_ch = sum(self.channels)
    
    def rot_ch_order_to_str(self):
        return ' '.join(list(map(lambda x: x + 'rotation', self.rot_ch_order)))

    def __str__(self):
        return self.definition_str
    
    def get_joint_index(self, joint_name):
        return self.joint_names.index(joint_name)

    def get_joint_names(self):
        return self.joint_names.copy()

    def get_global_joint_positions(self, frame):
        assert len(frame) == sum(self.channels)
        global_positions = [np.zeros(3) for _ in range(self.num_joints)]

        channel_idx = 0
        global_orientations = [np.eye(3) for _ in range(self.num_joints)]
        for i in range(self.num_joints):
            translation = np.zeros(3)
            euler_angles = np.zeros(3)
            if self.channels[i] == 3:
                euler_angles = np.array(frame[channel_idx:channel_idx+3])
                channel_idx += 3
            elif self.channels[i] == 6:
                translation = np.array(frame[channel_idx:channel_idx+3])
                euler_angles = np.array(frame[channel_idx+3:channel_idx+6])
                channel_idx += 6
            elif self.channels[i] == 0:
                pass
            else:
                raise NotImplementedError('Only 3 or 6 channels are supported')

            parent_idx = self.parent_idx[i]
            parent_position = global_positions[parent_idx] if parent_idx != -1 else np.zeros(3)
            parent_orientation = global_orientations[parent_idx] if parent_idx != -1 else np.eye(3)
            rotation = R.from_euler(self.rot_ch_order, euler_angles, degrees=True).as_matrix()
            global_orientations[i] = parent_orientation @ rotation
            global_positions[i] = parent_position + parent_orientation @ (translation + self.offsets[i])

        return global_positions

    def frame_to_rotvec(self, frame):
        assert len(frame) == sum(self.channels)
        rotvecs = []
        channel_idx = 0
        for i in range(self.num_joints):
            euler_angles = np.zeros(3)
            if self.channels[i] == 3:
                euler_angles = np.array(frame[channel_idx:channel_idx+3])
                channel_idx += 3
            elif self.channels[i] == 6:
                translation = np.array(frame[channel_idx:channel_idx+3])
                rotvecs.extend(translation)
                euler_angles = np.array(frame[channel_idx+3:channel_idx+6])
                channel_idx += 6
            elif self.channels[i] == 0:
                pass
            else:
                raise NotImplementedError('Only 3 or 6 channels are supported')

            rotvecs.extend(R.from_euler(self.rot_ch_order, euler_angles, degrees=True).as_rotvec())

        return rotvecs
    
    def rotvec_to_frame(self, rotvecs):
        assert len(rotvecs) == self.num_ch
        frame = []

        channel_idx = 0
        for i in range(self.num_joints):
            if self.channels[i] == 6:
                frame.extend(rotvecs[channel_idx:channel_idx+3])
                rotvec = rotvecs[channel_idx+3:channel_idx+6]
                frame.extend(R.from_rotvec(rotvec).as_euler(self.rot_ch_order, degrees=True))
            elif self.channels[i] == 3:
                rotvec = rotvecs[channel_idx:channel_idx+3]
                frame.extend(R.from_rotvec(rotvec).as_euler(self.rot_ch_order, degrees=True))
            channel_idx += self.channels[i]

        return frame

class BVH:
    def __init__(self, s: str):
        self.skeleton = None
        self.frames = []
        self.frame_time = 0.0
        self.parse(s)
    
    def parse(self, s:str):
        skel_str, motion_str = s.split('MOTION\n')

        self.skeleton = BVHSkeleton(skel_str)
        self._parse_frames(motion_str)
    
    def _parse_frames(self, s: str):
        lines = s.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                break
            if 'Frames:' in line:
                self.num_frames = int(line.split(' ')[1])
            elif 'Frame Time:' in line:
                self.frame_time = float(line.split(' ')[2])
            else:
                frame = line.split(' ')
                assert len(frame) == sum(self.skeleton.channels), f'Expected {sum(self.skeleton.channels)} channels, got {len(frame)}'
                self.frames.append(np.array(list(map(float, frame))))

    def __str__(self):
        s = str(self.skeleton)
        s += 'MOTION\n'
        s += f'Frames: {self.num_frames}\n'
        s += f'Frame Time: {self.frame_time}\n'
        for frame in self.frames:
            s += ' '.join(list(map(str, frame))) + '\n'
        return s

    def get_global_joint_positions(self, frame_idx):
        return self.skeleton.get_global_joint_positions(self.frames[frame_idx])

    def save(self, file_path):
        with open(file_path, 'w') as f:
            f.write(str(self))


def animate_3d_points(frames, title="3D Scatter Animation"):
    """
    Creates an animation of a list of 3D points over multiple frames.
    
    Parameters:
        frames (list of list of tuple): A list of frames, each containing a list of (x, y, z) coordinates.
        title (str): The title of the animation.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Get global axis limits from first and last frame
    all_points = frames[0] + frames[-1]
    x_vals, y_vals, z_vals = zip(*all_points)
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    z_min, z_max = min(z_vals), max(z_vals)
    
    # Ensure equal scaling for all axes
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
    mid_x = (x_max + x_min) / 2.0
    mid_y = (y_max + y_min) / 2.0
    mid_z = (z_max + z_min) / 2.0

    # Set view to make Y-axis vertical
    ax.view_init(elev=90, azim=-90)
    
    def update(frame_idx):
        ax.clear()
        x_vals, y_vals, z_vals = zip(*frames[frame_idx])
        ax.scatter(x_vals, y_vals, z_vals, c='b', marker='o')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title(title)
        
        # Fix the axis limits with equal aspect ratio
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=33, repeat=True)
    plt.show()


def animate_3d_points_for_compare(frames1, frames2, title="3D Scatter Animation"):
    """
    Creates an animation of a list of 3D points over multiple frames.
    
    Parameters:
        frames (list of list of tuple): A list of frames, each containing a list of (x, y, z) coordinates.
        title (str): The title of the animation.
    """
    assert len(frames1) == len(frames2)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Get global axis limits from first and last frame
    all_points = frames1[0] + frames1[-1]
    x_vals, y_vals, z_vals = zip(*all_points)
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    z_min, z_max = min(z_vals), max(z_vals)
    
    # Ensure equal scaling for all axes
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
    mid_x = (x_max + x_min) / 2.0
    mid_y = (y_max + y_min) / 2.0
    mid_z = (z_max + z_min) / 2.0

    # Set view to make Y-axis vertical
    ax.view_init(elev=90, azim=-90)
    
    def update(frame_idx):
        ax.clear()
        x_vals, y_vals, z_vals = zip(*frames1[frame_idx])
        ax.scatter(x_vals, y_vals, z_vals, c='b', marker='o')
        x_vals, y_vals, z_vals = zip(*frames2[frame_idx])
        ax.scatter(x_vals, y_vals, z_vals, c='r', marker='o')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title(title)
        
        # Fix the axis limits with equal aspect ratio
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ani = animation.FuncAnimation(fig, update, frames=len(frames1), interval=33, repeat=True)
    plt.show()


def test_bvh(path):
    with open(path, 'r') as f:
        s = f.read()
    skeleton = BVHSkeleton(s)
    print(skeleton.joint_names)
    print(skeleton.parent_idx)
    print(skeleton.offsets)
    print(skeleton.num_joints)
    print(skeleton.rot_ch_order)
    print(skeleton.channels)

    bvh = BVH(s)

    print(bvh.get_global_joint_positions(0))

    animate_3d_points([bvh.get_global_joint_positions(i) for i in range(bvh.num_frames)][:150])


def test_bvh_load_save():
    with open('lafan1_template.bvh', 'r') as f:
        s = f.read()
    bvh = BVH(s)
    bvh.save('lafan1_template_test.bvh')

    with open('lafan1_template_test.bvh', 'r') as f:
        s2 = f.read()
    bvh2 = BVH(s2)

    assert bvh.skeleton.definition_str == bvh2.skeleton.definition_str
    assert bvh.num_frames == bvh2.num_frames
    assert bvh.frame_time == bvh2.frame_time
    for i in range(bvh.num_frames):
        assert np.allclose(bvh.get_global_joint_positions(i), bvh2.get_global_joint_positions(i))


if __name__ == '__main__':
    # test_bvh_load_save()
    test_bvh('lafan1_template.bvh')
    # test_bvh('blanket_occlusion_to_lafan1.bvh')

