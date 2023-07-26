import numpy as np
from Motion.InverseKinematics import animation_from_positions
from Motion import BVH

def smpl2bvh(npy_file='results.npy', dataset='Humanact12'):
    motion_data = np.load(npy_file, allow_pickle=True)['motion']
    motion_data = motion_data.transpose(0, 3, 1, 2)

    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    SMPL_JOINT_NAMES = [
        'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
    ]

    if dataset == 'HumanML3D':
        parents = parents[:-2]
        SMPL_JOINT_NAMES = SMPL_JOINT_NAMES[:-2]

    bvh_path = npy_file[:-4] + 'anim{}.bvh'

    for i, p in enumerate(motion_data):
        print(f'starting anim no. {i}')
        anim, sorted_order, _ = animation_from_positions(p, parents)
        BVH.save(bvh_path.format(i), anim, names=np.array(SMPL_JOINT_NAMES)[sorted_order])

if __name__ == "__main__":
    smpl2bvh()