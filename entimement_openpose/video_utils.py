import os

import ffmpeg


def crop_video(input_path, output_dir, x1, y1, x2, y2):
    """Crops the given input file

    Parameters
    ----------
    input_path : str
        Path to input video
    x1 : int
        x position of top-left of cropped rectangle
    y1 : int
        y position of top-left of cropped rectangle
    x2 : int
        x position of bottom-right of cropped rectangle
    y2 : int
        y position of bottom-right of cropped rectangle

    Returns
    -------
    str
        Path of cropped video

    """
    output_filepath = os.path.join(output_dir, "cropped.mp4")
    ffmpeg.input(input_path).crop(x1, y1, x2 - x1, y2 - y1).output(
        output_filepath
    ).run()
    return output_filepath
