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

    def get_person_keypoints(self, person_index, parts=None, confidence_threshold=0, previous_body_keypoints_df=pd.DataFrame()):
        """Get the keypoints of a given person.

        Parameters
        ----------
        person_index : int
            Index of person in file for which to get keypoints
           
        parts : array of OpenPoseParts
            Array of parts to include in returned dataframe. Defaults to None,
            which shows all parts.
            
        confidence_threshold: float threshold in [0, 1] for confidence 
            Any keypoint candidate with lower confidence will be replaced by previous keypoint if that has a higher confidence. Default is an empty dataframe.    
            
        previous_body_keypoints_df: data frame of previous frame in video, if existent.
            Default is None.                

        Returns
        -------
        DataFrame
            DataFrame containing the keypoints

        """
        return self.get_multiple_keypoints([person_index], parts, confidence_threshold, previous_body_keypoints_df)

    def get_multiple_keypoints(self, person_indices, parts=None, confidence_threshold=0, previous_body_keypoints_df=pd.DataFrame()):
        """Get the keypoints of a given person.

        Parameters
        ----------
        person_indices : array of ints
            Indices of people in file for which to get keypoints

        parts : array of OpenPoseParts
            Array of parts to include in returned dataframe. Defaults to None,
            which shows all parts.
            
        confidence_threshold: float threshold in [0, 1] for confidence 
            Any keypoint candidate with lower confidence will be replaced by previous keypoint if that has a higher confidence. Default is an empty dataframe.    
            
        previous_body_keypoints_df: data frame of previous frame in video, if existent.
            Default is None.                

        Returns
        -------
        DataFrame
            DataFrame containing the keypoints, labelled with e.g. x0, y0,
            confidence0, x1, y1, confidence1

        """
        column_names = []
        body_keypoints_df = pd.DataFrame()

        for i in range(len(person_indices)):
            pi = person_indices[i]
            if pi < self.get_person_count():
                for c in self.COLUMN_NAMES:
                    column_names.append(c + str(i))

                person_keypoints = self.all_data['people'][pi]['pose_keypoints_2d']
                np_keypoints = np.array(person_keypoints)
                # Reshape to rows of x,y,confidence
                np_v_reshape = np_keypoints.reshape(int(len(np_keypoints)/3), 3)
		
                # Place in dataframe
                body_keypoints_df = pd.concat([body_keypoints_df,
                                               pd.DataFrame(np_v_reshape)],
                                              axis=1)

        body_keypoints_df.columns = column_names
        body_keypoints_df.index = self.ROW_NAMES
        # Check whether previous frame had higher confidence points and replace
        if not previous_body_keypoints_df.empty:
            for row in body_keypoints_df.itertuples():
            	cname = []
            	xname = []
            	yname = []	
            	for p in range(int(len(body_keypoints_df.columns)/3)):
            		cname = 'confidence'+str(p)  
            		xname = 'x'+str(p)
            		yname = 'y'+str(p)
            		try:
            			#print(body_keypoints_df.loc[row.Index, cname])
            			#print(previous_body_keypoints_df.loc[row.Index,
            			if body_keypoints_df.loc[row.Index, cname] < confidence_threshold and body_keypoints_df.loc[row.Index, cname] < previous_body_keypoints_df.loc[row.Index, cname]:
            				body_keypoints_df.loc[row.Index, xname] = previous_body_keypoints_df.loc[row.Index, xname]
            				body_keypoints_df.loc[row.Index, yname] = previous_body_keypoints_df.loc[row.Index, yname]
            				body_keypoints_df.loc[row.Index, cname] = previous_body_keypoints_df.loc[row.Index, cname]	 
            		except:
            			print("Skip", row.Index)


        if parts:
            part_names = [x.value for x in parts]
            body_keypoints_df = body_keypoints_df.loc[part_names]

        return body_keypoints_df
