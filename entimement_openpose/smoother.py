# import os
import pandas as pd
import cv2
import numpy as np
from scipy import signal

from .openpose_parts import OpenPoseParts


class Smoother:
    """Smoother for data frames created from OpenPose json files

    Parameters
    ----------
    smoothing window: int
        length of smoothing window, has to be odd (the higher the more frames are taken into account for smoothing; between 11 and 33 or so seems to work well, if it's too big it will affect/delay genuine movements)
    
    polyorder: int
       order of the polynomial used in fitting function, has to be smaller than the smoothing window (2 seems to work well, 1 connects with straight lines, etc.)

    Attributes
    ----------
    smoothing window: int
        length of smoothing window
    
    polyorder: int
       order of the polynomial used in smoothing function

    """

    def __init__(self, smoothing_window, polyorder):
        self.smoothing_window = smoothing_window
        self.polyorder = polyorder

    def smooth(self, body_keypoints_dfs):
        """Smooth keypoint positions over a number of frames
     	
        Parameters
        ----------
        body_keypoints_dfs: list of data frames with keypoints
    	
        Returns
        -------
        body_keypoints_dfs
        List of smoothed data frames
        """

        # Concatenate to one big dataframe all frames (assuming they are sorted wrt to person-order)
        big_body_keypoints_df = pd.concat(body_keypoints_dfs)
        num_frames = len(body_keypoints_dfs)
        num_people = int(len(big_body_keypoints_df.columns) / 3)
        num_bodyparts = len(
            body_keypoints_dfs[0]
        )  # assuming that we have the same number of bodyparts in each frame

        # I am sure there is a more pythonic way to do this, but I'll go for the loops now
        for this_bodypart in range(num_bodyparts):
            keypoints_series = big_body_keypoints_df.loc[
                big_body_keypoints_df.index[this_bodypart]
            ]
            keypoints_series_smoothed = signal.savgol_filter(
                keypoints_series, self.smoothing_window, self.polyorder, axis=0
            )
            big_body_keypoints_df.loc[
                big_body_keypoints_df.index[this_bodypart]
            ] = keypoints_series_smoothed

        for i in range(num_frames):
            body_keypoints_dfs[i] = big_body_keypoints_df.iloc[
                i * num_bodyparts : (i + 1) * num_bodyparts
            ]

        return body_keypoints_dfs
