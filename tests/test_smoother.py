import numpy as np
import pandas as pd

from raga_pose_estimation.reshaper import reshape_dataframes
from raga_pose_estimation.smoother import Smoother
from raga_pose_estimation.openpose_json_parser import OpenPoseJsonParser
from raga_pose_estimation.openpose_parts import OpenPoseParts

from . import dummy_dataframes


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

    person_dfs = reshape_dataframes(keypoint_array)

    # Smooth
    smoother = Smoother(5, 2)
    smoothed_dfs = smoother.smooth(person_dfs)

    # Check that nothing as changed, as you're smoothing over all the same frames
    for i, smoothed_df in enumerate(smoothed_dfs):
        pd.testing.assert_frame_equal(smoothed_df, person_dfs[i])

    # Build array where every other frame has only zeros
    zero_keypoints = pd.DataFrame(
        0, columns=sorted_keypoints.columns, index=sorted_keypoints.index
    )
    for i in range(0, 20, 2):
        keypoint_array[i] = zero_keypoints

    person_dfs = reshape_dataframes(keypoint_array)

    # Smooth
    smoother = Smoother(3, 1)
    smoothed_dfs = smoother.smooth(person_dfs)

    # As we are doing linear interpolation and the boundary mode is "interp" the two first and two last dataframes have to be equal
    for i, smoothed_df in enumerate(smoothed_dfs):
        for part in smoothed_df.columns.levels[0]:
            part_df = smoothed_df[part]
            if np.isnan(part_df["x"]).any():
                orig_part_df = person_dfs[i][part]
                # Rows with alternate NaNs and 0s won't be smoothed
                pd.testing.assert_frame_equal(part_df, orig_part_df)
            else:
                # As we are doing linear interpolation and the boundary mode is
                # "interp" the two first and two last dataframes have to be equal
                # (unless the original indices are NaN)
                pd.testing.assert_series_equal(
                    part_df.iloc[0], part_df.iloc[1], check_names=False
                )
                pd.testing.assert_series_equal(
                    part_df.iloc[-1], part_df.iloc[-2], check_names=False
                )

                # Values for the rest have to alternate
                for j in range(1, 19, 2):
                    assert (part_df.iloc[j] < part_df.iloc[j + 1]).all()


def test_chunk_and_smooth(dummy_dataframes):
    # Create an array based on the dummy dataframes
    keypoint_array = []
    for i in range(4):
        for j in range(4):
            keypoint_array.append(dummy_dataframes[0])
            keypoint_array.append(dummy_dataframes[1])
        keypoint_array.append(dummy_dataframes[2])
        keypoint_array.append(dummy_dataframes[2])

    person_dfs = reshape_dataframes(keypoint_array)

    smoother = Smoother(5, 2)
    smoothed_dfs = smoother.smooth(person_dfs)

    # Smoothed ear dataframe:
    # x flips between 2, 3, 4
    # y flips between 8, 5, 4
    # c flips between 0.1, 0.5, 0.7
    # Smoothed values will be around these ranges (plus a bit extra)
    smoothed_ear_df = smoothed_dfs[0][OpenPoseParts.R_EAR.value]
    assert smoothed_ear_df["x"].between(2, 4.3).all()
    assert smoothed_ear_df["y"].between(3.4, 8).all()
    assert smoothed_ear_df["c"].between(0.1, 0.8).all()

    # Smooth eye dataframe (includes NaNs):
    # x flips between 15, 10, NaN
    # y flips between 5, 4, NaN
    # c flips between 0.8, 0.9, NaN
    # Nans are at indices 8, 9, 18, 19...
    smoothed_eye_df = smoothed_dfs[0][OpenPoseParts.R_EYE.value]
    for index, row in smoothed_eye_df.iterrows():
        if index % 10 > 7:
            assert np.isnan(row).all()
        else:
            assert row["x"] >= 10 and row["x"] <= 15
            assert row["y"] >= 4 and row["y"] <= 5
            assert row["c"] >= 0.8 and row["c"] <= 0.9
