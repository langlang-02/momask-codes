"""SOMA Uniform to HumanML3D 22-joint mapping.

The target order follows MoMask/HumanML3D:
0 pelvis, 1 left hip, 2 right hip, 3 spine, 4 left knee, 5 right knee,
6 chest, 7 left ankle, 8 right ankle, 9 upper chest, 10 left foot,
11 right foot, 12 neck/head, 13 left shoulder, 14 right shoulder,
15 head, 16 left arm, 17 right arm, 18 left forearm, 19 right forearm,
20 left hand, 21 right hand.
"""

HML3D_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine",
    "left_knee",
    "right_knee",
    "chest",
    "left_ankle",
    "right_ankle",
    "upper_chest",
    "left_foot",
    "right_foot",
    "neck",
    "left_shoulder",
    "right_shoulder",
    "head",
    "left_arm",
    "right_arm",
    "left_forearm",
    "right_forearm",
    "left_hand",
    "right_hand",
]


# Each target joint lists candidate SOMA BVH joint names in priority order.
# If the BONES-SEED skeleton uses slightly different names, edit this file.
SOMA_TO_HML3D = {
    0: ["Hips", "Pelvis", "Root"],
    1: ["LeftUpLeg", "LeftUpperLeg", "LeftThigh", "LeftLeg"],
    2: ["RightUpLeg", "RightUpperLeg", "RightThigh", "RightLeg"],
    3: ["Spine1", "Spine"],
    4: ["LeftShin", "LeftLowerLeg", "LeftKnee", "LeftLeg"],
    5: ["RightShin", "RightLowerLeg", "RightKnee", "RightLeg"],
    6: ["Spine2", "Chest"],
    7: ["LeftFoot", "LeftAnkle"],
    8: ["RightFoot", "RightAnkle"],
    9: ["Chest", "UpperChest", "Spine3"],
    10: ["LeftToeBase", "LeftToe", "LeftFoot"],
    11: ["RightToeBase", "RightToe", "RightFoot"],
    12: ["Neck1", "Neck", "Head"],
    13: ["LeftShoulder"],
    14: ["RightShoulder"],
    15: ["Head", "HeadEnd"],
    16: ["LeftArm", "LeftUpperArm"],
    17: ["RightArm", "RightUpperArm"],
    18: ["LeftForeArm", "LeftForearm", "LeftLowerArm"],
    19: ["RightForeArm", "RightForearm", "RightLowerArm"],
    20: ["LeftHand", "LeftWrist"],
    21: ["RightHand", "RightWrist"],
}


# HumanML3D preprocessing constants used by utils.motion_process.process_file.
HML3D_FACE_JOINTS = [2, 1, 17, 16]  # right hip, left hip, right shoulder, left shoulder
HML3D_LEFT_LEG = (5, 8)
HML3D_FOOT_RIGHT = [8, 11]
HML3D_FOOT_LEFT = [7, 10]
