
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
