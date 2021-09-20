from enum import Enum


class OpenPoseParts(Enum):
    """
    Enum with body parts in order defined by OpenPose.
    (We could get these from OpenPose, but defining them here means we can work
    on machines where OpenPose is not installed.)
    """

    NOSE = "Nose"
    NECK = "Neck"
    R_SHOULDER = "RShoulder"
    R_ELBOW = "RElbow"
    R_WRIST = "RWrist"
    L_SHOULDER = "LShoulder"
    L_ELBOW = "LElbow"
    L_WRIST = "LWrist"
    MID_HIP = "MidHip"
    R_HIP = "RHip"
    R_KNEE = "RKnee"
    R_ANKLE = "RAnkle"
    L_HIP = "LHip"
    L_KNEE = "LKnee"
    L_ANKLE = "LAnkle"
    R_EYE = "REye"
    L_EYE = "LEye"
    R_EAR = "REar"
    L_EAR = "LEar"
    L_BIG_TOE = "LBigToe"
    L_SMALL_TOE = "LSmallToe"
    L_HEEL = "LHeel"
    R_BIG_TOE = "RBigToe"
    R_SMALL_TOE = "RSmallToe"
    R_HEEL = "RHeel"


class OpenPosePartGroups:
    UPPER_BODY_PARTS = [
        OpenPoseParts.NOSE,
        OpenPoseParts.NECK,
        OpenPoseParts.R_SHOULDER,
        OpenPoseParts.R_ELBOW,
        OpenPoseParts.R_WRIST,
        OpenPoseParts.L_SHOULDER,
        OpenPoseParts.L_ELBOW,
        OpenPoseParts.L_WRIST,
        OpenPoseParts.R_EYE,
        OpenPoseParts.L_EYE,
        OpenPoseParts.R_EAR,
        OpenPoseParts.L_EAR,
        OpenPoseParts.MID_HIP,
    ]

    LOWER_BODY_PARTS = [
        OpenPoseParts.MID_HIP,
        OpenPoseParts.R_HIP,
        OpenPoseParts.R_KNEE,
        OpenPoseParts.R_ANKLE,
        OpenPoseParts.L_HIP,
        OpenPoseParts.L_KNEE,
        OpenPoseParts.L_ANKLE,
        OpenPoseParts.L_BIG_TOE,
        OpenPoseParts.L_SMALL_TOE,
        OpenPoseParts.L_HEEL,
        OpenPoseParts.R_BIG_TOE,
        OpenPoseParts.R_SMALL_TOE,
        OpenPoseParts.R_HEEL,
    ]


# Jin modification
# and HandPoseParts


class HandPoseParts(Enum):
    """
    Enum with body parts in order defined by OpenPose.
    (We could get these from OpenPose, but defining them here means we can work
    on machines where OpenPose is not installed.)
    """

    L_WRIST = "LWrist"
    L_THUMB1 = "LThumb1"
    L_THUMB2 = "LThumb2"
    L_THUMB3 = "LThumb3"
    L_THUMB4 = "LThumb4"
    L_INDEX1 = "LIndex1"
    L_INDEX2 = "LIndex2"
    L_INDEX3 = "LIndex3"
    L_INDEX4 = "LIndex4"
    L_MIDDLE1 = "LMiddle1"
    L_MIDDLE2 = "LMiddle2"
    L_MIDDLE3 = "LMiddle3"
    L_MIDDLE4 = "LMiddle4"
    L_RING1 = "LRing1"
    L_RING2 = "LRing2"
    L_RING3 = "LRing3"
    L_RING4 = "LRing4"
    L_LITTLE1 = "LLittle1"
    L_LITTLE2 = "LLittle2"
    L_LITTLE3 = "LLittle3"
    L_LITTLE4 = "LLittle4"

    R_WRIST = "RWrist"
    R_THUMB1 = "RThumb1"
    R_THUMB2 = "RThumb2"
    R_THUMB3 = "RThumb3"
    R_THUMB4 = "RThumb4"
    R_INDEX1 = "RIndex1"
    R_INDEX2 = "RIndex2"
    R_INDEX3 = "RIndex3"
    R_INDEX4 = "RIndex4"
    R_MIDDLE1 = "RMiddle1"
    R_MIDDLE2 = "RMiddle2"
    R_MIDDLE3 = "RMiddle3"
    R_MIDDLE4 = "RMiddle4"
    R_RING1 = "RRing1"
    R_RING2 = "RRing2"
    R_RING3 = "RRing3"
    R_RING4 = "RRing4"
    R_LITTLE1 = "RLittle1"
    R_LITTLE2 = "RLittle2"
    R_LITTLE3 = "RLittle3"
    R_LITTLE4 = "RLittle4"
