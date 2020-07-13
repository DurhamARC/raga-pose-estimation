#import os
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
        length of smoothing window
    
    polyorder: int
       order of the polynomial used in fitting function, has to be smaller than the smoothing window

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
        
        # Go through all frames (have to be sorted)
        # TODO: Do we assume it is already sorted or sort it here?	
        # Das Beispiel hatte alle keypoints in einem Dataframe (mit nur einer Person), und dann pro Keypointart (was wohl eine Liste gibt) geglaettet. Wir haben eine Liste von Dataframes, potentiell mit mehrern Personen. Wir machen wir das da am Schlauesten...?
        big_body_keypoints_df = pd.concat(body_keypoints_dfs)  
        num_people = int(len(big_body_keypoints_df.columns)/3)
       
        big_body_keypoints_df.index.name='bodyparts'
        
        #big_body_keypoints_df = big_body_keypoints_df.groupby('bodyparts')#.apply(list)
        
        big_body_keypoints_df = big_body_keypoints_df.transpose()
        
        big_body_keypoints_df = pd.DataFrame({i: big_body_keypoints_df[i].values.T.ravel() for i in set(big_body_keypoints_df.columns)})
        
        print(big_body_keypoints_df)
        
        big_body_keypoints_df = pd.DataFrame(signal.savgol_filter(big_body_keypoints_df, self.smoothing_window, self.polyorder, axis=0), columns = big_body_keypoints_df.columns, index = big_body_keypoints_df.index)
        
        print(big_body_keypoints_df)
               
        
        
        # Go through all persons and keypoints
        #for p in range(num_people):
        #    for i in range
            #cname = "confidence" + str(p)
            #xname = "x" + str(p)
            #yname = "y" + str(p)
            #big_body_keypoints_df_x = big_body_keypoints_df.loc[OpenPoseParts.MID_HIP.value].groupby(xname)
            #print(big_body_keypoints_df_x.loc[xname])
        #print(signal.savgol_filter(big_body_keypoints_df, self.smoothing_window, self.polyorder))
            #body_keypoints_dfs[yname, i] = signal.savgol_filter(body_keypoints_dfs[yname], self.smoothing_window, self.polyorder)
    	
        return body_keypoints_dfs
