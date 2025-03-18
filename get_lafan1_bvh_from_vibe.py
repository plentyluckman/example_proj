import numpy as np
from scipy.optimize import minimize
import joblib
import os
from tqdm import tqdm
from utils_bvh import *

def get_spin_joint_names():
    # vibe 3d joint indices
    return [
        'OP Nose',        # 0
        'OP Neck',        # 1
        'OP RShoulder',   # 2
        'OP RElbow',      # 3
        'OP RWrist',      # 4
        'OP LShoulder',   # 5
        'OP LElbow',      # 6
        'OP LWrist',      # 7
        'OP MidHip',      # 8
        'OP RHip',        # 9
        'OP RKnee',       # 10
        'OP RAnkle',      # 11
        'OP LHip',        # 12
        'OP LKnee',       # 13
        'OP LAnkle',      # 14
        'OP REye',        # 15
        'OP LEye',        # 16
        'OP REar',        # 17
        'OP LEar',        # 18
        'OP LBigToe',     # 19
        'OP LSmallToe',   # 20
        'OP LHeel',       # 21
        'OP RBigToe',     # 22
        'OP RSmallToe',   # 23
        'OP RHeel',       # 24
        'rankle',         # 25
        'rknee',          # 26
        'rhip',           # 27
        'lhip',           # 28
        'lknee',          # 29
        'lankle',         # 30
        'rwrist',         # 31
        'relbow',         # 32
        'rshoulder',      # 33
        'lshoulder',      # 34
        'lelbow',         # 35
        'lwrist',         # 36
        'neck',           # 37
        'headtop',        # 38
        'hip',            # 39 'Pelvis (MPII)', # 39
        'thorax',         # 40 'Thorax (MPII)', # 40
        'Spine (H36M)',   # 41
        'Jaw (H36M)',     # 42
        'Head (H36M)',    # 43
        'nose',           # 44
        'leye',           # 45 'Left Eye', # 45
        'reye',           # 46 'Right Eye', # 46
        'lear',           # 47 'Left Ear', # 47
        'rear',           # 48 'Right Ear', # 48
    ]

name_idx_map = {
    'Hips': 39,

    'LeftUpLeg': 28,
    'LeftLeg': 13, 
    'LeftFoot': 21,
    'LeftToe': 19,
    
    'RightUpLeg': 27,
    'RightLeg': 10,
    'RightFoot': 24,
    'RightToe': 22,
    
    'Spine1': 41,
    'Neck': 37,
    'Head': 43,
    
    'LeftArm': 5,
    'LeftForeArm': 6,
    'LeftHand': 7,

    'RightArm': 2,
    'RightForeArm': 3,
    'RightHand': 4,
    }

joints_ik_weights = {
    'LeftUpLeg': 2., 
    'LeftLeg': 1., 
    'LeftFoot': 1., 
    'LeftToe': 1.,
    'RightUpLeg': 2., 
    'RightLeg': 1., 
    'RightFoot': 1., 
    'RightToe': 1.,
    'Spine1': 1.,
    'Neck': 1.,
    'Head': 1.,
    'RightArm': 1., 
    'RightForeArm': 1., 
    'RightHand': 1.,
    'LeftArm': 1., 
    'LeftForeArm': 1., 
    'LeftHand': 1.
}

VIBE_TO_LAFAN1_SCALE = 100
LAFAN1_TO_VIBE_SCALE = 1. / VIBE_TO_LAFAN1_SCALE

def objective(x, lafan1_skel, vibe_pos):
    vibe_hips_pos = vibe_pos[name_idx_map['Hips']]
    root_offset = -np.array(lafan1_skel.offsets[0]) + vibe_hips_pos * VIBE_TO_LAFAN1_SCALE
    
    q = np.zeros(len(x) + 3)
    q[:3] = root_offset
    q[3:] = x

    sums = 0
    
    global_positions = lafan1_skel.get_global_joint_positions(lafan1_skel.rotvec_to_frame(q))
    for joint_name, joint_weight in joints_ik_weights.items():
        joint_idx = lafan1_skel.get_joint_index(joint_name)
        global_position = global_positions[joint_idx] * LAFAN1_TO_VIBE_SCALE
        sums += joint_weight * np.linalg.norm(global_position - vibe_pos[name_idx_map[joint_name]]) ** 2

    sums += 0.000001 * (np.linalg.norm(x) ** 2)

    return sums


def ik_vibe(lafan1_skel: BVHSkeleton, joints_3d, x0):
    rot_vibe = R.from_rotvec([np.pi, 0, 0]).as_matrix()
    vibe_pos = [(rot_vibe @ joint_3d) for joint_3d in joints_3d]
    vibe_hips_pos = vibe_pos[name_idx_map['Hips']]
    root_offset = -np.array(lafan1_skel.offsets[0]) + vibe_hips_pos * VIBE_TO_LAFAN1_SCALE

    ik_res = minimize(objective, x0, args=(lafan1_skel, vibe_pos))
    q_ = np.zeros(lafan1_skel.num_ch)
    q_[:3] = root_offset
    q_[3:] = ik_res.x

    return ik_res.x, lafan1_skel.rotvec_to_frame(q_)


def main(input_vibe_file, input_bvh_file, output_bvh_file):
    with open(input_bvh_file, 'r') as f:
        bvh = BVH(f.read())
    skel = bvh.skeleton

    data = joblib.load(input_vibe_file)
    print(len(data))
    print(data.keys())

    joint_3d_infos = data[1]['joints3d']

    frame_ids = list(map(int, data[1]['frame_ids']))

    frame_num = int(np.max(frame_ids))+1
    missing_ranges = []
    for _i in range(len(frame_ids)-1):
        if frame_ids[_i+1] - frame_ids[_i] > 1:
            missing_ranges.append((frame_ids[_i], frame_ids[_i+1]))


    joint_num = len(joint_3d_infos[0])

    print("frame_num:", frame_num)
    print("joint_num:", joint_num)
    print("missing ranges: ", missing_ranges)
    print(joint_3d_infos.shape)

    missing_frames = []

    new_bvh = BVH(str(bvh))
    new_bvh.frames = []
    new_bvh.num_frames = 0
    
    x0 = np.zeros(skel.num_ch-3)
    for frame_idx in tqdm(range(frame_num)):
        if not (frame_idx in frame_ids):
            missing_frames.append(frame_idx)
            continue

        data_frame = frame_ids.index(frame_idx)
        x, frame = ik_vibe(skel, joint_3d_infos[data_frame].copy(), x0)
        new_bvh.frames.append(frame)
        new_bvh.num_frames += 1

        # use the last frame as the initial guess for the next frame
        x0 = x.copy()
    
    new_bvh.save(output_bvh_file)


def animate_vibe_and_bvh(input_vibe_file, input_bvh_file):
    with open(input_bvh_file, 'r') as f:
        bvh = BVH(f.read())

    data = joblib.load(input_vibe_file)

    # frame_ids = list(map(int, data[1]['frame_ids']))
    frame_ids = list(range(bvh.num_frames))

    missing_ranges = []
    for _i in range(len(frame_ids)-1):
        if frame_ids[_i+1] - frame_ids[_i] > 1:
            missing_ranges.append((frame_ids[_i], frame_ids[_i+1]))

    joint_3d_infos = data[1]['joints3d']

    rot_vibe = R.from_rotvec([np.pi, 0, 0]).as_matrix()
    all_frames_vibe = [[(rot_vibe @ joint_3d_info) for joint_3d_info in joint_3d_infos[i]] for i in range(len(frame_ids))]

    bvh_global_positions = [np.array(bvh.get_global_joint_positions(i)) * LAFAN1_TO_VIBE_SCALE for i in range(len(frame_ids))]

    animate_3d_points_for_compare(bvh_global_positions, all_frames_vibe)


if __name__ == '__main__':
    main(os.path.join('data', 'blanket_occlusion.pkl'), os.path.join('lafan1_template.bvh'), 'blanket_occlusion_to_lafan1.bvh')
    animate_vibe_and_bvh(os.path.join('data', 'blanket_occlusion.pkl'), 'blanket_occlusion_to_lafan1.bvh')
