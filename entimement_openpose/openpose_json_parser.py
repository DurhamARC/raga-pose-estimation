import json
import numpy as np
import pandas as pd

from .openpose_parts import OpenPoseParts


class OpenPoseJsonParser:
    """Parser for JSON files created by OpenPose

    Parameters
    ----------
    filepath : string
        Full path to the JSON file to parse

    Attributes
    ----------
    all_data : type
        Dictionary of values from JSON file
    COLUMN_NAMES : type
        Names of columns in resulting data frame
    ROW_NAMES : type
        Names of rows in resulting data frame

    """

    COLUMN_NAMES = ['x', 'y', 'confidence']
    ROW_NAMES = [p.value for p in OpenPoseParts]

    def __init__(self, filepath):
        with open(filepath) as f:
            self.all_data = json.load(f)

    def get_person_count(self):
        """Get the count of the people in the file.

        Returns
        -------
        int
            Number of people detected in file

        """
        return len(self.all_data['people'])

    def get_person_keypoints(self, person_index):
        """Get the keypoints of a given person.

        Parameters
        ----------
        person_index : int
            Index of person in file for which to get keypoints

        Returns
        -------
        DataFrame
            DataFrame containing the keypoints

        """
        if person_index < self.get_person_count():
            person_keypoints = self.all_data['people'][person_index]['pose_keypoints_2d']
            np_keypoints = np.array(person_keypoints)
            # Reshape to rows of x,y,confidence
            np_v_reshape = np_keypoints.reshape(int(len(np_keypoints)/3), 3)
            # Place in dataframe
            body_keypoints_df = pd.DataFrame(np_v_reshape,
                                             columns=self.COLUMN_NAMES,
                                             index=self.ROW_NAMES)
            return body_keypoints_df
        else:
            return None
