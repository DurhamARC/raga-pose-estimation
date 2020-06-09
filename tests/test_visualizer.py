import math
import os

import numpy as np
import pandas as pd
import pytest

from entimement_openpose.openpose_parts import OpenPoseParts
from entimement_openpose.visualizer import Visualizer


@pytest.fixture
def dummy_dataframe():
    data = {
        'x': [2, 15],
        'y': [8, 5],
        'confidence': [0.1, 0.8]
    }
    df = pd.DataFrame(data,
                      columns=['x', 'y', 'confidence'],
                      index=pd.Index([OpenPoseParts.R_EAR.value,
                                      OpenPoseParts.R_EYE.value]))
    return df


def test_create_video(dummy_dataframe):
    viz = Visualizer('output')

    # Ensure file doesn't exist
    output_filename = 'output/test_blank.avi'
    if os.path.exists(output_filename):
        os.remove(output_filename)

    # Create a video from the dataframe on a blank background
    viz.create_videos_from_dataframes('test', [dummy_dataframe], 20, 10)

    # Check video file has been created and is about the size we expect
    assert(os.path.isfile('output/test_blank.avi'))
    size = os.path.getsize(output_filename)
    assert(size > 1000 and size < 10000)


def test_create_overlay_invalid():
    viz = Visualizer('output')

    # Create overlay without specifying video to overlay
    with pytest.raises(ValueError):
        viz.create_videos_from_dataframes('test_invalid', [pd.DataFrame()],
                                          20, 10, create_overlay=True)

    # Try to create overlay specifying invalid video
    with pytest.raises(ValueError):
        viz.create_videos_from_dataframes('test_invalid', [pd.DataFrame()],
                                          20, 10, create_overlay=True,
                                          video_to_overlay="notarealvideo.avi")

    # Try to create overlay with valid file but not enough frames
    many_dfs = [pd.DataFrame() for x in range(500)]
    with pytest.raises(ValueError):
        viz.create_videos_from_dataframes('test_invalid', many_dfs,
                                          20, 10, create_overlay=True,
                                          video_to_overlay="example_files/short_video.mp4")


def test_draw_points(dummy_dataframe):
    viz = Visualizer('output')
    height = 10
    width = 20
    img = np.ones((height, width, 3), np.uint8)
    viz.draw_points({'test': img}, dummy_dataframe)

    # Points from data frame in x, y coordinates
    expected_points = [(2, 8), (15, 5)]

    # Check pixels im modified img array
    for y in range(0, height):
        for x in range(0, width):
            # Check how far given pixel is from a point to be drawn
            min_distance = min([math.sqrt((ex - x)**2 + (ey - y)**2)
                                for (ex, ey) in expected_points])

            # Pixels close to the point should be blue (BGR array)
            if min_distance <= 3:
                assert(img[y, x].tolist() == [255, 0, 0])
            # Pixels further away should be black (we don't check distance = 4
            # as the results vary for our inaccurate distance method)
            else:
                assert(img[y, x].tolist() == [1, 1, 1])


def test_draw_lines(dummy_dataframe):
    viz = Visualizer('output')
    height = 10
    width = 20
    img = np.ones((height, width, 3), np.uint8)
    viz.draw_lines({'test': img}, dummy_dataframe, viz.LINE_PATHS)

    # Points from data frame in x, y coordinates
    expected_points = [(2, 8), (15, 5)]

    # Check pixels in modified img array
    # Expected points should be white
    for (ex, ey) in expected_points:
        assert(img[ey, ex].tolist() == [255, 255, 255])

    # Pixels about half way between should also be white
    assert(img[6, 8].tolist() == [255, 255, 255])
    assert(img[7, 9].tolist() == [255, 255, 255])

    # Random pixels away from the points should be black
    for (x, y) in [(0, 0), (9, 0), (19, 0), (0, 3), (9, 3), (19, 9)]:
        assert(img[y, x].tolist() == [1, 1, 1])
