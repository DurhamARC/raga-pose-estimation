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

    def smooth(self, person_dfs):
        """Smooth keypoint positions over a number of frames

        Parameters
        ----------
        person_dfs: list of data frames as returned by reshape_dataframes

        Returns
        -------
        body_keypoints_dfs
        List of smoothed data frames
        """
        smoothed_dfs = []
        for person_df in person_dfs:
            # Create copy so we don't modify original
            smoothed_df = person_df.copy()
            # print("before:")
            # print(person_df)
            # Split into parts so we can split into chunks where each part
            # appears/disappears, and smooth over each chunk separately
            for part in smoothed_df.columns.levels[0]:
                part_df = smoothed_df[part]
                nan_indices = np.where(part_df["x"].isna())[0]
                # print(f"Nan indices: {nan_indices}")
                smoothed_df[part] = part_df.apply(
                    lambda x: self._chunk_and_smooth_col(x, nan_indices)
                )

            smoothed_dfs.append(smoothed_df)
            # print("after:")
            # print(person_df)

        return smoothed_dfs

    def _chunk_and_smooth_col(self, col, nan_indices):
        smoothed_arrays = []

        if len(nan_indices) > 0:
            # We've got some null values, so need to split the values
            # at the nan indices
            split_col = np.split(col, nan_indices)
            for i, s in enumerate(split_col):
                # First value in each split after the first is NaN, so remove it
                if s.size > 0 and np.isnan(s.iloc[0]):
                    smoothed_arrays.append(pd.Series(s.iloc[0]))
                    s = s.iloc[1:]

                # Smooth if possible
                # Should we smooth with a smaller window rather than
                # not smoothing at all?
                if len(s) > self.smoothing_window:
                    s = signal.savgol_filter(
                        s, self.smoothing_window, self.polyorder
                    )

                smoothed_arrays.append(pd.Series(s))

            smoothed_col = pd.concat(smoothed_arrays, ignore_index=True)

            return smoothed_col

        else:
            return signal.savgol_filter(
                col, self.smoothing_window, self.polyorder
            )
