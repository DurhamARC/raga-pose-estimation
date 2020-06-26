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
