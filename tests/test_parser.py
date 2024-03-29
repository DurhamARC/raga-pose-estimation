import pandas as pd

from raga_pose_estimation.openpose_json_parser import OpenPoseJsonParser
from raga_pose_estimation.openpose_parts import (
    OpenPoseParts,
    OpenPosePartGroups,
)


def test_parser():
    parser = OpenPoseJsonParser(
        "example_files/example_3people/output_json/video_000000000093_keypoints.json"
    )  # choose one where the people are not already sorted
    assert parser.get_person_count() == 3

    # Check get_person_keypoints
    person_keypoints = parser.get_person_keypoints(1)
    assert type(person_keypoints) == pd.DataFrame
    assert person_keypoints.shape == (len(OpenPoseParts), 3)
    assert list(person_keypoints.columns) == parser.COLUMN_NAMES

    # Check getting multiple people
    all_keypoints = parser.get_multiple_keypoints([0, 1])
    assert type(all_keypoints) == pd.DataFrame
    assert all_keypoints.shape == (len(OpenPoseParts), 6)
    assert list(all_keypoints.columns) == [
        "x0",
        "y0",
        "confidence0",
        "x1",
        "y1",
        "confidence1",
    ]

    # Check that values in person_keypoints are the same as second set of
    # columns in all_keypoints (apart from column names)
    all_keypoints_person1 = all_keypoints.iloc[:, 3:6]
    all_keypoints_person1.columns = person_keypoints.columns
    assert all_keypoints_person1.equals(person_keypoints)

    # Check getting only upper parts
    upper_keypoints = parser.get_person_keypoints(
        1, OpenPosePartGroups.UPPER_BODY_PARTS
    )
    assert type(upper_keypoints) == pd.DataFrame
    assert upper_keypoints.shape == (
        len(OpenPosePartGroups.UPPER_BODY_PARTS),
        3,
    )
    assert OpenPoseParts.L_ANKLE not in upper_keypoints.index

    # Test person ordering (0 is left-most, 1 is next)
    sorted_person_keypoints = parser.sort_persons_by_x_position(all_keypoints)
    assert (
        sorted_person_keypoints.loc[OpenPoseParts.MID_HIP.value].iloc[0]
        < sorted_person_keypoints.loc[OpenPoseParts.MID_HIP.value].iloc[3]
    )

    # Test handing in a confidence threshold (and make sure it replaces values by the same values)
    sorted_person_keypoints2 = parser.get_multiple_keypoints(
        [0, 1], None, 0.7, sorted_person_keypoints
    )
    assert sorted_person_keypoints.equals(sorted_person_keypoints2)
