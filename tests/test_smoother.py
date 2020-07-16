import pandas as pd
from numpy.testing import assert_array_almost_equal, assert_allclose

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
    print(sorted_keypoints)
    for i in range(20):
        print(keypoint_array[i])
        pd.testing.assert_frame_equal(keypoint_array[i], sorted_keypoints)
