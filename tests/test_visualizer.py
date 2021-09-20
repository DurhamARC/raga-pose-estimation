import math
import os

import numpy as np
import pandas as pd
import pytest

from raga_pose_estimation.openpose_parts import OpenPoseParts
from raga_pose_estimation.visualizer import Visualizer
from . import single_frame_person_df, three_frame_person_dfs


def test_create_video(three_frame_person_dfs):
    viz = Visualizer("output")

    # Ensure file doesn't exist
    output_filename = "output/test_blank.mp4"
    if os.path.exists(output_filename):
        os.remove(output_filename)

    # Create a video from the dataframe on a blank background
    viz.create_video_from_dataframes("test", three_frame_person_dfs, 20, 10)

    # Check video file has been created and is about the size we expect
    assert os.path.isfile("output/test_blank.mp4")
    size = os.path.getsize(output_filename)
    assert size > 1000 and size < 10000


def test_create_overlay_invalid():
    viz = Visualizer("output")

    # Create overlay without specifying video to overlay
    with pytest.raises(ValueError):
        viz.create_video_from_dataframes(
            "test_invalid", [pd.DataFrame()], 20, 10, create_overlay=True
        )

    # Try to create overlay specifying invalid video
    with pytest.raises(ValueError):
        viz.create_video_from_dataframes(
            "test_invalid",
            [pd.DataFrame()],
            20,
            10,
            create_overlay=True,
            video_to_overlay="notarealvideo.avi",
        )

    # Try to create overlay with valid file but not enough frames
    many_dfs = [pd.DataFrame() for x in range(500)]
    with pytest.raises(ValueError):
        viz.create_video_from_dataframes(
            "test_invalid",
            many_dfs,
            20,
            10,
            create_overlay=True,
            video_to_overlay="example_files/short_video.mp4",
        )


def test_draw_points(single_frame_person_df):
    viz = Visualizer("output")
    height = 10
    width = 20
    img = np.ones((height, width, 3), np.uint8)
    viz.draw_points(img, single_frame_person_df, 0)

    # Points from data frame in x, y coordinates
    expected_points = [(2, 8), (15, 5)]

    # Check pixels im modified img array
    for y in range(0, height):
        for x in range(0, width):
            # Check how far given pixel is from a point to be drawn
            min_distance = min(
                [
                    math.sqrt((ex - x) ** 2 + (ey - y) ** 2)
                    for (ex, ey) in expected_points
                ]
            )

            # Pixels close to the point should be blue (BGR array)
            if min_distance <= 3:
                assert img[y, x].tolist() == [255, 0, 0]
            # Pixels further away should be black (we don't check distance = 4
            # as the results vary for our inaccurate distance method)
            else:
                assert img[y, x].tolist() == [1, 1, 1]


def test_draw_lines(single_frame_person_df):
    viz = Visualizer("output")
    height = 10
    width = 20
    img = np.ones((height, width, 3), np.uint8)
    viz.draw_lines(img, single_frame_person_df, 0, viz.LINE_PATHS)

    # Points from data frame in x, y coordinates
    expected_points = [(2, 8), (15, 5)]

    # Check pixels in modified img array
    # Expected points should be white
    for (ex, ey) in expected_points:
        assert img[ey, ex].tolist() == [255, 255, 255]

    # Pixels about half way between should also be white
    assert img[6, 8].tolist() == [255, 255, 255]
    assert img[7, 9].tolist() == [255, 255, 255]

    # Random pixels away from the points should be black
    for (x, y) in [(0, 0), (9, 0), (19, 0), (0, 3), (9, 3), (19, 9)]:
        assert img[y, x].tolist() == [1, 1, 1]
