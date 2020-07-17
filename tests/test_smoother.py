import pandas as pd

from entimement_openpose.smoother import Smoother
from entimement_openpose.openpose_json_parser import OpenPoseJsonParser
from entimement_openpose.openpose_parts import (
    OpenPoseParts,
    OpenPosePartGroups,
)


def test_smoother():
    parser = OpenPoseJsonParser(
        "example_files/example_3people/output_json/video_000000000093_keypoints.json"
    )

    # Get the keypoints
    keypoints = parser.get_multiple_keypoints([0, 1])
    # Ensure person ordering (0 is left-most, 1 is next)
    sorted_keypoints = parser.sort_persons_by_x_position(keypoints)
    # Create an array that has all the same frames
    keypoint_array = []
    for i in range(20):
        keypoint_array.append(sorted_keypoints)

    # Smooth
    smoother = Smoother(5, 2)
    keypoint_array = smoother.smooth(keypoint_array)

    # Check that nothing as changed, as you're smoothing over all the same frames
    for i in range(20):
        pd.testing.assert_frame_equal(keypoint_array[i], sorted_keypoints)

    # Build array where every other frame has only zeros
    zero_keypoints = pd.DataFrame(
        0, columns=sorted_keypoints.columns, index=sorted_keypoints.index
    )
    for i in range(0, 20, 2):
        keypoint_array[i] = zero_keypoints

    smoother = Smoother(3, 1)
    keypoint_array = smoother.smooth(keypoint_array)

    # As we are doing linear interpolation and the boundary mode is "interp" the two first and two last dataframes have to be equal
    pd.testing.assert_frame_equal(keypoint_array[0], keypoint_array[1])
    pd.testing.assert_frame_equal(keypoint_array[-1], keypoint_array[-2])

    # Values for the rest have to alternate
    for i in range(1, 19, 2):
        assert keypoint_array[i].iloc[0, 0] < keypoint_array[i + 1].iloc[0, 0]
        assert keypoint_array[i].iloc[5, 1] < keypoint_array[i + 1].iloc[5, 1]
        assert keypoint_array[i].iloc[7, 3] < keypoint_array[i + 1].iloc[7, 3]
