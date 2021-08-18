import os

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from .openpose_parts import OpenPoseParts


### could add signal light when creating the video ####


class Visualizer:
    """Class providing visualization of OpenPose data from a DataFrame"""

    MID_COLOR = (0, 0, 255)
    L_COLOR = (0, 255, 0)
    R_COLOR = (255, 0, 0)
    LINE_COLOR = (255, 255, 255)

    LINE_PATHS = [
        [OpenPoseParts.NOSE, OpenPoseParts.NECK, OpenPoseParts.MID_HIP],
        [
            OpenPoseParts.L_EAR,
            OpenPoseParts.L_EYE,
            OpenPoseParts.NOSE,
            OpenPoseParts.R_EYE,
            OpenPoseParts.R_EAR,
        ],
        [
            OpenPoseParts.NECK,
            OpenPoseParts.L_SHOULDER,
            OpenPoseParts.L_ELBOW,
            OpenPoseParts.L_WRIST,
        ],
        [
            OpenPoseParts.NECK,
            OpenPoseParts.R_SHOULDER,
            OpenPoseParts.R_ELBOW,
            OpenPoseParts.R_WRIST,
        ],
        [OpenPoseParts.L_HIP, OpenPoseParts.MID_HIP, OpenPoseParts.R_HIP],
        [
            OpenPoseParts.L_HIP,
            OpenPoseParts.L_KNEE,
            OpenPoseParts.L_ANKLE,
            OpenPoseParts.L_BIG_TOE,
        ],
        [
            OpenPoseParts.R_HIP,
            OpenPoseParts.R_KNEE,
            OpenPoseParts.R_ANKLE,
            OpenPoseParts.R_BIG_TOE,
        ],
        [OpenPoseParts.L_ANKLE, OpenPoseParts.L_SMALL_TOE],
        [OpenPoseParts.R_ANKLE, OpenPoseParts.R_SMALL_TOE],
    ]

    def __init__(self, output_directory=""):
        """Initializes a Visualizer instance with a set of parts to display and
        an output directory.

        Parameters
        ----------
        output_directory : str
            Path to video output folder. Directory will be created if it
            doesn't exist.

        Returns
        -------
        Visualizer instance

        """
        self.output_directory = output_directory
        if not os.path.exists(self.output_directory):
            os.mkdir(output_directory)

    def draw_points(self, img, person_dfs, frame_index):
        """Draws keypoints on to the given image.

        Parameters
        ----------
        img : array of np.array
            Array of image arrays in OpenCV format

        person_dfs : list of DataFrame
            DataFrame containing person keypoints

        frame_index: int
            index of frame to draw

        Returns
        -------
        None

        """
        for person_df in person_dfs:
            row = person_df.iloc[frame_index]

            for part in row.index.levels[0]:
                if not np.isnan(row[part]).any():
                    pos = (int(row[part]["x"]), int(row[part]["y"]))

                    color = Visualizer.MID_COLOR
                    if part.startswith("R"):
                        color = Visualizer.R_COLOR
                    elif part.startswith("L"):
                        color = Visualizer.L_COLOR

                    img = cv2.circle(img, pos, 5, color, -1)

    def draw_lines(self, img, person_dfs, frame_index, paths):
        """Draws lines joining body parts on to the given image arrays.

        Parameters
        ----------
        imgs : dictionary of np.array
            Dictionary of image arrays in OpenCV format

        person_dfs : list of DataFrame
            DataFrame containing person keypoints

        frame_index: int
            index of frame to draw

        paths: array of arrays of OpenPoseParts
            arrays of paths to display, where each path is an array of
            OpenPoseParts

        Returns
        -------
        None

        """
        for person_df in person_dfs:
            row = person_df.iloc[frame_index]
            for line in paths:
                pts = np.zeros(shape=(len(line), 2), dtype=np.int32)
                count = 0

                for part in line:
                    if part.value in row.index.levels[0]:
                        # if not np.isnan(row[part.value]).any():
                        if not (row[part.value] <= 0).any():

                            pt = np.int32(row[part.value].values)
                            pts[count] = pt[:2]
                            count += 1

                # Filter out zero points, then reshape before drawing
                pts = pts[pts > 0]
                pts = pts.reshape((-1, 1, 2))

                cv2.polylines(
                    img,
                    [pts],
                    False,
                    Visualizer.LINE_COLOR,
                    thickness=4,
                )

    def get_paths_from_dataframe(person_df):
        paths = []

        for path in Visualizer.LINE_PATHS:
            new_path = []
            for part in path:
                if part.value in person_df.columns.levels[0]:
                    new_path.append(part)
            if new_path:
                paths.append(new_path)

        return paths

    # Jin modification

    def create_video_from_dataframes(
        self,
        file_basename,
        person_dfs,
        width,
        height,
        create_overlay=False,
        video_to_overlay=None,
        intervals=None,
        accumulate=False,
    ):
        """Creates a video visualising the provided array of body keypoints.
        The lines to draw will be determined by the parts in the first
        dataframe.

        Parameters
        ----------
        file_basename : str
            Base name of file. Will create files <file_basename>_blank.avi
            and/or <file_basename>_overlay.avi
        person_dfs : array of DataFrames
            Array of DataFrames as created by reshape_dataframes
        width : int
            Width of output video
        height : type
            Height of output video
        create_overlay : bool
            Whether to create the visualisation on top of an overlay
            (default False)
        video_to_overlay : str
            Path to video to overlay. Must be provided if create_overlay is
            True
        interval: list example:[[start1, end1], [s2, e2], [s3, e3]]
            The list of a time interval including start and end time.
            Drawing a green circle in the top-left of the frame that included
            in the interval, otherwise drawing a red circle. If default, do not
            draw anything.
            (default None)
        accumulate: bool
            If true, retain the previous skeletons
            (default False)


        Returns
        -------
        None

        """
        cap = None
        if create_overlay:
            if not video_to_overlay:
                raise ValueError(
                    "You must provide video_to_overlay "
                    "if create_overlay is True"
                )

            if not os.path.exists(video_to_overlay):
                raise ValueError("video_to_overlay is not a valid file")

            cap = cv2.VideoCapture(video_to_overlay)

        paths = Visualizer.get_paths_from_dataframe(person_dfs[0])

        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        name = "overlay" if create_overlay else "blank"
        filename = os.path.join(
            self.output_directory, "%s_%s.mp4" % (file_basename, name)
        )
        out = cv2.VideoWriter(filename, fourcc, 25, (width, height))

        # Draw the data from the DataFrame
        if intervals is not None:
            rest_list = []
            for interval in intervals:
                rest_list += list(range(interval[0], interval[1]))

        if (not create_overlay) and accumulate:
            img_acc = np.zeros((height, width, 3), np.uint8)

        for i in range(len(person_dfs[0].index)):
            # if i > 10000:
            #     continue
            if i % 1000 == 0:
                print(
                    "{}/{} frames are saved".format(
                        i, len(person_dfs[0].index)
                    )
                )
            if create_overlay:
                ret, frame = cap.read()
                if not ret:
                    break

                img = frame
            else:
                img = np.zeros((height, width, 3), np.uint8)

            if (not create_overlay) and accumulate:
                self.draw_points(img, person_dfs, i)
                img_acc = cv2.add(img_acc, img)
            else:
                self.draw_points(img, person_dfs, i)
                self.draw_lines(img, person_dfs, i, paths)

            if intervals is not None:
                if i in rest_list:
                    cv2.circle(img, (50, 50), 15, (0, 0, 255), -1)
                else:
                    cv2.circle(img, (50, 50), 15, (0, 255, 0), -1)

            if (not create_overlay) and accumulate:
                out.write(img_acc)
            else:
                out.write(img)

        if create_overlay:
            cap.release()

        out.release()
