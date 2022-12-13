import os

import ffmpeg


def crop_video(input_path, output_dir, w, h, x, y):
    """Crops the given input file

    Parameters
    ----------
    input_path : str
        Path to input video
    w : int
        width of cropped rectangle
    h : int
        height of cropped rectangle
    x : int
        x position of top-left of cropped rectangle
    y : int
        y position of top-left of cropped rectangle

    Returns
    -------
    str
        Path of cropped video

    """
    output_filepath = os.path.join(output_dir, "cropped.mp4")
    if w < 1 or h < 1:
        raise ValueError("Crop width and height must be greater than 0")

    if x < 0 or y < 0:
        raise ValueError(
            "Crop rectangle coordinates must be greater than or equal to 0"
        )

    ffmpeg.input(input_path).crop(x, y, w, h).output(output_filepath).run()
    return output_filepath

