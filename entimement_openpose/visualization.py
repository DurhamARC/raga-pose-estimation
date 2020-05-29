import cv2
import numpy as np

from .openpose_parts import OpenPoseParts


class Visualization:
    """Class providing visualization of OpenPose data from a DataFrame"""

    MID_COLOR = (0, 0, 255)
    L_COLOR = (0, 255, 0)
    R_COLOR = (255, 0, 0)
    LINE_COLOR = (255, 255, 255)

    LINE_PATHS = [
        [OpenPoseParts.NOSE, OpenPoseParts.NECK, OpenPoseParts.MID_HIP],
        [OpenPoseParts.L_EAR, OpenPoseParts.L_EYE, OpenPoseParts.NOSE,
            OpenPoseParts.R_EYE, OpenPoseParts.R_EAR],
        [OpenPoseParts.NECK, OpenPoseParts.L_SHOULDER, OpenPoseParts.L_ELBOW,
            OpenPoseParts.L_WRIST],
        [OpenPoseParts.NECK, OpenPoseParts.R_SHOULDER, OpenPoseParts.R_ELBOW,
            OpenPoseParts.R_WRIST],
        [OpenPoseParts.L_HIP, OpenPoseParts.MID_HIP, OpenPoseParts.R_HIP]
    ]

    def draw_points(img, pt_df):
        """Draws keypoints on to the given image array.

        Parameters
        ----------
        img : np.array
            Image array in OpenCV format

        pt_df : DataFrame
            DataFrame containing keypoints

        Returns
        -------
        np.array
            Image array in OpenCV format

        """
        for index, row in pt_df.iterrows():
            pos = (int(row['x']), int(row['y']))

            color = Visualization.MID_COLOR
            if row.name.startswith('R'):
                color = Visualization.R_COLOR
            elif row.name.startswith('L'):
                color = Visualization.L_COLOR

            if pos[0] > 0 or pos[1] > 0:
                img = cv2.circle(img, pos, 3, color, -1)

    def draw_lines(img, pt_df):
        """Draws lines joining body parts on to the given image array.

        Parameters
        ----------
        img : np.array
            Image array in OpenCV format

        pt_df : DataFrame
            DataFrame containing keypoints

        Returns
        -------
        np.array
            Image array in OpenCV format

        """
        for line in Visualization.LINE_PATHS:
            pts = np.zeros(shape=(len(line), 2), dtype=np.int32)
            count = 0

            for part in line:
                row = pt_df.loc[part.value, 'x':'y']
                pt = np.int32(row.values)
                if pt[0] > 0 and pt[1] > 0:
                    pts[count] = pt
                    count += 1

            # Filter out zero points, then reshape before drawing
            pts = pts[pts > 0]
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], False, Visualization.LINE_COLOR,
                          thickness=2)

    def create_video_from_dataframes(filename, body_keypoints_dfs, width, height):
        """Creates a video visualising the provided array of body keypoints.

        Parameters
        ----------
        filename : str
            Path to output file (should be .mp4)
        body_keypoints_dfs : array of DataFrames
            Array of DataFrames as created by OpenPoseJsonParser
        width : int
            Width of output video
        height : type
            Height of output video

        Returns
        -------
        None

        """
        # Draw the data from the DataFrame
        img_array = []
        for df in body_keypoints_dfs:
            img = np.ones((height, width, 3), np.uint8)
            Visualization.draw_lines(img, df)
            Visualization.draw_points(img, df)
            img_array.append(img)

        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 25,
                              (width, height))
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
