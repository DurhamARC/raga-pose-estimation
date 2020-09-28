import os

import cv2
import pytest

from entimement_openpose.video_utils import crop_video


def test_crop_video():
    # Ensure file doesn't exist
    output_filename = os.path.join("output", "cropped.mp4")
    if os.path.exists(output_filename):
        os.remove(output_filename)

    # Create a video from the dataframe on a blank background
    w = 600
    h = 300
    x = 200
    y = 300
    output_path = crop_video(
        "example_files/example_1person/short_video.mp4", "output", w, h, x, y,
    )
    assert output_path == output_filename

    # Check video file has been created
    assert os.path.isfile(output_filename)

    # Check that video size is as we expect
    cap = cv2.VideoCapture(output_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    assert width == w
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    assert height == h
    cap.release()


def test_invalid_crop_params():
    test_params = [
        ((0, 5, 10, 20), "Crop width and height must be greater than 0"),
        ((2, -1, 10, 20), "Crop width and height must be greater than 0"),
        (
            (20, 20, -1, 5),
            "Crop rectangle coordinates must be greater than or equal to 0",
        ),
        (
            (20, 20, 1, -5),
            "Crop rectangle coordinates must be greater than or equal to 0",
        ),
    ]

    for params, expected in test_params:
        with pytest.raises(ValueError) as excinfo:
            crop_video(
                "example_files/example_1person/short_video.mp4",
                "output",
                *params,
            )
        assert expected in str(excinfo.value)
