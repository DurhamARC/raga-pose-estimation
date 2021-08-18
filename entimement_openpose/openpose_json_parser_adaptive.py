import json
import numpy as np
import pandas as pd

from .openpose_parts import OpenPoseParts

### adaptive detect the number of singer in the video #####

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

    COLUMN_NAMES = ["x", "y", "confidence"]
    ROW_NAMES = [p.value for p in OpenPoseParts]

    def __init__(self, filepath):
        print(filepath.split('/')[-1])
        with open(filepath) as f:
            self.all_data = json.load(f)
        print('{} persons are detected'.format(len(self.all_data['people'])))

    def get_person_count(self):
        """Get the count of the people in the file.

        Returns
        -------
        int
            Number of people detected in file

        """
        return len(self.all_data["people"])

    def get_person_keypoints(
        self,
        person_index,
        parts=None,
        confidence_threshold=0,
        previous_body_keypoints_df=None,
    ):
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
            DataFrame containing the keypoints, labelled as x, y, confidence

        """
        df = self.get_multiple_keypoints([person_index], parts)
        df.columns = self.COLUMN_NAMES
        return df

    def sort_persons_by_x_position(self, body_keypoints_df):
        """Sort the data so that the left-most person has index 0, the next has index 1, etc.

        Parameters
        ----------
        body_keypoints_df: data frame that is to be sorted.

        Returns
        -------
        sorted_body_keypoints_df
            Sorted DataFrame.
        """
        sorted_body_keypoints_df = pd.DataFrame()

        # Find permutation to sort x values in ascending order
        idx = np.argsort(body_keypoints_df.mean(axis=0).iloc[0::3])

        # Test whether permutation equals just the numbering (1, 2, 3, ...), i.e. whether x values are already sorted.
        if list(range(len(idx))) == list(idx):
            sorted_body_keypoints_df = sorted_body_keypoints_df.join(
                body_keypoints_df, how="right"
            )
        # Otherwise, sort them
        else:
            for i in range(len(idx)):
                cname = "confidence" + str(i)
                cname_old = "confidence" + str(idx[i])
                xname = "x" + str(i)
                xname_old = "x" + str(idx[i])
                yname = "y" + str(i)
                yname_old = "y" + str(idx[i])

                sorted_body_keypoints_df = sorted_body_keypoints_df.join(
                    body_keypoints_df[[xname_old]], how="right", rsuffix="new"
                )
                sorted_body_keypoints_df = sorted_body_keypoints_df.join(
                    body_keypoints_df[[yname_old]], how="right", rsuffix="new"
                )
                sorted_body_keypoints_df = sorted_body_keypoints_df.join(
                    body_keypoints_df[[cname_old]], how="right", rsuffix="new"
                )

                sorted_body_keypoints_df.rename(
                    columns={sorted_body_keypoints_df.columns[i * 3]: xname},
                    inplace=True,
                )
                sorted_body_keypoints_df.rename(
                    columns={
                        sorted_body_keypoints_df.columns[i * 3 + 1]: yname
                    },
                    inplace=True,
                )
                sorted_body_keypoints_df.rename(
                    columns={
                        sorted_body_keypoints_df.columns[i * 3 + 2]: cname
                    },
                    inplace=True,
                )

        return sorted_body_keypoints_df

    def get_multiple_keypoints(
        self,
        person_indices,
        parts=None,
        confidence_threshold=0,
        previous_body_keypoints_df=None,
    ):
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
        ### Jin modification
        ### The person_indeces is removed
        ### We can automatically match the skeleton in current frame with that in the previous frame
        
        
        column_names = []
        body_keypoints_df = pd.DataFrame()

        for i in range(len(self.all_data['people'])):

            person_keypoints = self.all_data["people"][i][
                "pose_keypoints_2d"
            ]
            np_keypoints = np.array(person_keypoints)
            # Reshape to rows of x,y,confidence
            np_v_reshape = np_keypoints.reshape(
                int(len(np_keypoints) / 3), 3
            )

            # Place in dataframe
            person_df = pd.DataFrame(np_v_reshape)

            # Check if x, y, confidence are all 0, and replace with nulls
            
            def replace_zeros(row):
                if (row == 0).all():
                    return np.nan
                else:
                    return row
            
            if (person_df.iloc[0]==0).all():
                person_df.iloc[0] = np.nan
                
            person_df = person_df.apply(
                lambda row: replace_zeros(row), axis=1
            )

            body_keypoints_df = pd.concat(
                [body_keypoints_df, person_df], axis=1
            )
        
        
        
        for p in range(int(len(body_keypoints_df.columns) / 3)):
            cname = "confidence" + str(p)
            xname = "x" + str(p)
            yname = "y" + str(p)
            column_names.append(xname)
            column_names.append(yname)
            column_names.append(cname)
        
        if len(body_keypoints_df) == 0 and not previous_body_keypoints_df is None:
            body_keypoints_df = previous_body_keypoints_df
        else:            
            body_keypoints_df.columns = column_names
            body_keypoints_df.index = self.ROW_NAMES
        
        body_keypoints_df = match_previous_df(body_keypoints_df, previous_body_keypoints_df, parts=parts)
        
        # Check whether previous frame had higher confidence points and replace
        if (
            not previous_body_keypoints_df is None
            and not previous_body_keypoints_df.empty
        ):
            for row in body_keypoints_df.itertuples():

                for p in range(int(len(body_keypoints_df.columns) / 3)):
                    cname = "confidence" + str(p)
                    xname = "x" + str(p)
                    yname = "y" + str(p)

                    if row.Index in previous_body_keypoints_df.index:

                        body_keypoints_df = self.sort_persons_by_x_position(
                            body_keypoints_df
                        )

                        if (
                            body_keypoints_df.loc[row.Index, cname]
                            < confidence_threshold
                            and body_keypoints_df.loc[row.Index, cname]
                            < previous_body_keypoints_df.loc[row.Index, cname]
                        ):
                            body_keypoints_df.loc[
                                row.Index, xname
                            ] = previous_body_keypoints_df.loc[
                                row.Index, xname
                            ]
                            body_keypoints_df.loc[
                                row.Index, yname
                            ] = previous_body_keypoints_df.loc[
                                row.Index, yname
                            ]
                            body_keypoints_df.loc[
                                row.Index, cname
                            ] = previous_body_keypoints_df.loc[
                                row.Index, cname
                            ]

        if parts:
            part_names = [x.value for x in parts]
            body_keypoints_df = body_keypoints_df.loc[part_names]

        return body_keypoints_df

## Jin modification
def match_previous_df(body_keypoints_df, previous_body_keypoints_df, parts=None, distance_thres=50):
    '''

    Parameters
    ----------
    body_keypoints_df : dataframe
        data frame of current frame in video, if existent.
            Default is None.
            
    previous_body_keypoints_df : dataframe
        data frame of previous frame in video, if existent.
            Default is None.
            
    parts : array of OpenPoseParts
            Array of parts to include in returned dataframe. Defaults to None,
            which shows all parts.
            
    distance_thres : int
        the threshold of mean distance to identify two skeletons into the same person. 
        The default is 50.

    Returns
    -------
    body_keypoints_df : TYPE
        DESCRIPTION.

    '''
    # if detection is unrealable (too much NaN), then delete this person
    if previous_body_keypoints_df is None:
        for name in body_keypoints_df:
            if len(np.where(np.isnan(body_keypoints_df[name]))[0]) >= 20:
                body_keypoints_df = body_keypoints_df.drop(columns=[name])
        
    # rename the columns according to current number of persons
        column_names = []        
        for p in range(int(len(body_keypoints_df.columns) / 3)):
            cname = "confidence" + str(p)
            xname = "x" + str(p)
            yname = "y" + str(p)
            column_names.append(xname)
            column_names.append(yname)
            column_names.append(cname)
        body_keypoints_df.columns = column_names
        
            
    
    # if the detected person is more than previous (anchor), compute best matches 
    if (
        not previous_body_keypoints_df is None
        and len(body_keypoints_df.keys()) != len(previous_body_keypoints_df.keys())
    ):
        if parts:
            part_names = [x.value for x in parts]
            body_keypoints_df = body_keypoints_df.loc[part_names]
        body_keypoints_df_new = pd.DataFrame()
        for p in range(int(len(previous_body_keypoints_df.columns) / 3)):
            dist_list = []
            xname = "x" + str(p)
            yname = "y" + str(p)
            person1 = np.array(previous_body_keypoints_df[[xname,yname]])
            for q in range(int(len(body_keypoints_df.columns) / 3)):
                xname = "x" + str(q)
                yname = "y" + str(q)
                person2 = np.array(body_keypoints_df[[xname,yname]])
                dist_list.append(mean_body_distance(person1, person2))
            current_person_id = np.argmin(dist_list)
            
            if dist_list[current_person_id] < distance_thres:
                new_person_id = int(len(body_keypoints_df_new.columns)/3)
                body_keypoints_df_new["x"+str(new_person_id)] = \
                    body_keypoints_df["x"+str(current_person_id)]
                body_keypoints_df_new["y"+str(new_person_id)] = \
                    body_keypoints_df["y"+str(current_person_id)]
                body_keypoints_df_new["confidence"+str(new_person_id)] = \
                    body_keypoints_df["confidence"+str(current_person_id)]
                
            else:
                new_person_id = int(len(body_keypoints_df_new.columns)/3)
                body_keypoints_df_new["x"+str(new_person_id)] = \
                    previous_body_keypoints_df["x"+str(p)]
                body_keypoints_df_new["y"+str(new_person_id)] = \
                    previous_body_keypoints_df["y"+str(p)]
                body_keypoints_df_new["confidence"+str(new_person_id)] = \
                    previous_body_keypoints_df["confidence"+str(p)]
        
        column_names = []
        for p in range(int(len(body_keypoints_df_new.columns) / 3)):
            cname = "confidence" + str(p)
            xname = "x" + str(p)
            yname = "y" + str(p)
            column_names.append(xname)
            column_names.append(yname)
            column_names.append(cname)
        body_keypoints_df_new.columns = column_names
        body_keypoints_df = body_keypoints_df_new
    
    return body_keypoints_df
                                   

def mean_body_distance(person1, person2):
    '''

    Parameters
    ----------
    person1 : 25x2 np.array
    person2 : 25x2 np.array

    Returns
    -------
    distance : np.float 

    '''
    distances = np.sum((person1-person2)**2, axis=1)**0.5
    num_idx = np.where(~np.isnan(distances))[0]
    distance = np.mean(distances[num_idx])
    return distance
    
                    
                    
                    
                
