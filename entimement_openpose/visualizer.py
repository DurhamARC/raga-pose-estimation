import os

import cv2
import numpy as np

from .openpose_parts import OpenPoseParts


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

    def draw_points(self, img, pt_df):
        """Draws keypoints on to the given image.

        Parameters
        ----------
        img : array of np.array
            Array of image arrays in OpenCV format

        pt_df : DataFrame
            DataFrame containing keypoints

        Returns
        -------
        None

        """
        n_people = len(pt_df.columns) // 3
        for index, row in pt_df.iterrows():
            for i in range(n_people):
                pos = (int(row[i * 3]), int(row[i * 3 + 1]))

                color = Visualizer.MID_COLOR
                if row.name.startswith("R"):
                    color = Visualizer.R_COLOR
                elif row.name.startswith("L"):
                    color = Visualizer.L_COLOR

                if pos[0] > 0 or pos[1] > 0:
                    img = cv2.circle(img, pos, 3, color, -1)

    def draw_lines(self, img, pt_df, paths):
        """Draws lines joining body parts on to the given image arrays.

        Parameters
        ----------
        imgs : dictionary of np.array
            Dictionary of image arrays in OpenCV format

        pt_df : DataFrame
            DataFrame containing keypoints

        paths: array of arrays of OpenPoseParts
            arrays of paths to display, where each path is an array of
            OpenPoseParts

        Returns
        -------
        None

        """
        n_people = len(pt_df.columns) // 3

        for i in range(n_people):
            for line in paths:
                pts = np.zeros(shape=(len(line), 2), dtype=np.int32)
                count = 0

                for part in line:
                    if part.value in pt_df.index:
                        part_index = pt_df.index.get_loc(part.value)
                        row = pt_df.iloc[part_index, i * 3 : i * 3 + 2]
                        pt = np.int32(row.values)
                        if pt[0] > 0 and pt[1] > 0:
                            pts[count] = pt
                            count += 1

                # Filter out zero points, then reshape before drawing
                pts = pts[pts > 0]
                pts = pts.reshape((-1, 1, 2))

                cv2.polylines(
                    img, [pts], False, Visualizer.LINE_COLOR, thickness=2,
                )

    def get_paths_from_dataframe(df):
        paths = []

        for path in Visualizer.LINE_PATHS:
            new_path = []
            for part in path:
                if part.value in df.index:
                    new_path.append(part)
            if new_path:
                paths.append(new_path)

        return paths

    def create_video_from_dataframes(
        self,
        file_basename,
        body_keypoints_dfs,
        width,
        height,
        create_overlay=False,
        video_to_overlay=None,
    ):
        """Creates a video visualising the provided array of body keypoints.
        The lines to draw will be determined by the parts in the first
        dataframe.

        Parameters
        ----------
        file_basename : str
            Base name of file. Will create files <file_basename>_blank.avi
            and/or <file_basename>_overlay.avi
        body_keypoints_dfs : array of DataFrames
            Array of DataFrames as created by OpenPoseJsonParser
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

        paths = Visualizer.get_paths_from_dataframe(body_keypoints_dfs[0])

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        name = "overlay" if create_overlay else "blank"
        filename = os.path.join(
            self.output_directory, "%s_%s.avi" % (file_basename, name)
        )
        out = cv2.VideoWriter(filename, fourcc, 25, (width, height))

        # Draw the data from the DataFrame
        for df in body_keypoints_dfs:
            if create_overlay:
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(
                        "Not enough frames in overlay video to "
                        "match data frame"
                    )
                img = frame
            else:
                img = np.ones((height, width, 3), np.uint8)

            self.draw_lines(img, df, paths)
            self.draw_points(img, df)

            out.write(img)

        if create_overlay:
            cap.release()

        out.release()
