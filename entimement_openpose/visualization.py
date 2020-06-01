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

    def draw_points(imgs, pt_df):
        """Draws keypoints on to the given image arrays.

        Parameters
        ----------
        img : array of np.array
            Array of image arrays in OpenCV format

        pt_df : DataFrame
            DataFrame containing keypoints

        Returns
        -------
        np.array
            Image array in OpenCV format

        """
        n_people = len(pt_df. columns) // 3
        for index, row in pt_df.iterrows():
            for i in range(n_people):
                pos = (int(row['x'+str(i)]), int(row['y'+str(i)]))

                color = Visualization.MID_COLOR
                if row.name.startswith('R'):
                    color = Visualization.R_COLOR
                elif row.name.startswith('L'):
                    color = Visualization.L_COLOR

                if pos[0] > 0 or pos[1] > 0:
                    for key, img in imgs.items():
                        if img is not None:
                            img = cv2.circle(img, pos, 3, color, -1)

    def draw_lines(imgs, pt_df):
        """Draws lines joining body parts on to the given image arrays.

        Parameters
        ----------
        imgs : array of np.array
            Array of image arrays in OpenCV format

        pt_df : DataFrame
            DataFrame containing keypoints

        Returns
        -------
        np.array
            Image array in OpenCV format

        """
        n_people = len(pt_df.columns) // 3
        for i in range(n_people):
            for line in Visualization.LINE_PATHS:
                pts = np.zeros(shape=(len(line), 2), dtype=np.int32)
                count = 0

                for part in line:
                    row = pt_df.loc[part.value, 'x'+str(i):'y'+str(i)]
                    pt = np.int32(row.values)
                    if pt[0] > 0 and pt[1] > 0:
                        pts[count] = pt
                        count += 1

                # Filter out zero points, then reshape before drawing
                pts = pts[pts > 0]
                pts = pts.reshape((-1, 1, 2))

                for key, img in imgs.items():
                    if img is not None:
                        cv2.polylines(img, [pts], False,
                                      Visualization.LINE_COLOR, thickness=2)

    def create_videos_from_dataframes(directory, file_basename,
                                      body_keypoints_dfs, width, height,
                                      create_blank=True, create_overlay=False,
                                      video_to_overlay=None):
        """Creates a video visualising the provided array of body keypoints.

        Parameters
        ----------
        directory : str
            Path to output folder
        file_basename : str
            Base name of file. Will create files <file_basename>_blank.mp4
            and/or <file_basename>_overlay.mp4
        body_keypoints_dfs : array of DataFrames
            Array of DataFrames as created by OpenPoseJsonParser
        width : int
            Width of output video
        height : type
            Height of output video
        create_blank : bool
            Whether to create a visualisation with an empty background
            (default True)
        create_overlay : bool
            Whether to create a visualisation on top of an overlay
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
            try:
                cap = cv2.VideoCapture(video_to_overlay)
            except Exception as e:
                raise Exception("Unable to open video from %s"
                                % video_to_overlay) from e

        # Draw the data from the DataFrame
        img_arrays = {'blank': [], 'overlay': []}
        for df in body_keypoints_dfs:
            imgs = {'blank': None, 'overlay': None}

            if create_blank:
                imgs['blank'] = np.ones((height, width, 3), np.uint8)

            if create_overlay:
                if cap.isOpened():
                    ret, frame = cap.read()
                    imgs['overlay'] = frame
                else:
                    raise Exception("Not enough frames in overlay video to \
                                     match data frame")

            Visualization.draw_lines(imgs, df)
            Visualization.draw_points(imgs, df)

            for k, v in imgs.items():
                if v is not None:
                    img_arrays[k].append(v)

        if create_overlay:
            cap.release()

        for key, img_array in img_arrays.items():
            if len(img_array):
                out = cv2.VideoWriter("%s_%s.mp4" % (file_basename, key),
                                      cv2.VideoWriter_fourcc(*'avc1'), 25,
                                      (width, height))
                for i in range(len(img_array)):
                    out.write(img_array[i])

                out.release()
