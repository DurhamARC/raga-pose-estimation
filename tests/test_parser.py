import pandas as pd

from entimement_openpose.openpose_json_parser import OpenPoseJsonParser
from entimement_openpose.openpose_parts import OpenPoseParts, OpenPosePartGroups


def test_parser():
    parser = OpenPoseJsonParser('example_files/example_3people/output_json/video_000000000000_keypoints.json')
    assert(parser.get_person_count() == 3)

    # Check get_person_keypoints
    person_keypoints = parser.get_person_keypoints(1)
    assert(type(person_keypoints) == pd.DataFrame)
    assert(person_keypoints.shape == (len(OpenPoseParts), 3))
    assert(list(person_keypoints.columns) == parser.COLUMN_NAMES)

    # Check getting multiple people
    all_keypoints = parser.get_multiple_keypoints([0, 1])
    assert(type(all_keypoints) == pd.DataFrame)
    assert(all_keypoints.shape == (len(OpenPoseParts), 6))
    assert(list(all_keypoints.columns) == ['x0', 'y0', 'confidence0',
                                           'x1', 'y1', 'confidence1'])

    # Check that values in person_keypoints are the same as second set of
    # columns in all_keypoints (apart from column names)
    all_keypoints_person1 = all_keypoints.iloc[:, 3:6]
    all_keypoints_person1.columns = person_keypoints.columns
    assert(all_keypoints_person1.equals(person_keypoints))

    # Check getting only upper parts
    upper_keypoints = parser.get_person_keypoints(1, OpenPosePartGroups.UPPER_BODY_PARTS)
    assert(type(upper_keypoints) == pd.DataFrame)
    assert(upper_keypoints.shape == (len(OpenPosePartGroups.UPPER_BODY_PARTS), 3))
    assert(OpenPoseParts.L_ANKLE not in upper_keypoints.index)
