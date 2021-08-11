import math
import cv2
import numpy as np
import pandas as pd


##################################################################
##                                                              ##
## create a 3d skeleton video in a blank background             ##
##                                                              ##
##################################################################


# {0,  "Nose"}
# {1,  "Neck"},
# {2,  "RShoulder"},
# {3,  "RElbow"},
# {4,  "RWrist"},
# {5,  "LShoulder"},
# {6,  "LElbow"},
# {7,  "LWrist"},
# {8, "REye"},
# {9, "LEye"},
# {10, "MidHip"},
PARTS = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist',
         'LShoulder', 'LElbow', 'LWrist', 'REye', 'LEye', 'MidHip']
SKELETON_EDGES = np.array([[0, 1], [1, 2], [2, 3], [3, 4], 
                               [1, 5], [5, 6], [6, 7],
                               [1, 10], [8, 0], [9, 0]])

# theta, phi = 3.1415/4, -3.1415/6
theta, phi = -0.3, 0.24
should_rotate = False
scale_dx = 800
scale_dy = 800

# plot 3d skeleton
class Plotter3d:
    
    def __init__(self, canvas_size, origin=(0.5, 0.5), scale=1, parts=PARTS, skeleton_edges=SKELETON_EDGES):
        self.origin = np.array([origin[1] * canvas_size[1], origin[0] * canvas_size[0]], dtype=np.float32)  # x, y
        self.scale = np.float32(scale)
        self.theta = 0
        self.phi = 0
        self.parts = parts
        self.skeleton_edges = skeleton_edges
        axis_length = 200
        axes = [
            np.array([[-axis_length/2, -axis_length/2, 0], [axis_length/2, -axis_length/2, 0]], dtype=np.float32),
            np.array([[-axis_length/2, -axis_length/2, 0], [-axis_length/2, axis_length/2, 0]], dtype=np.float32),
            np.array([[-axis_length/2, -axis_length/2, 0], [-axis_length/2, -axis_length/2, axis_length]], dtype=np.float32)]
        step = 20
        for step_id in range(axis_length // step + 1):  # add grid
            axes.append(np.array([[-axis_length / 2, -axis_length / 2 + step_id * step, 0],
                                  [axis_length / 2, -axis_length / 2 + step_id * step, 0]], dtype=np.float32))
            axes.append(np.array([[-axis_length / 2 + step_id * step, -axis_length / 2, 0],
                                  [-axis_length / 2 + step_id * step, axis_length / 2, 0]], dtype=np.float32))
        self.axes = np.array(axes)

    def plot(self, img, vertices, edges):
        global theta, phi
        img.fill(0)
        R = self._get_rotation(theta, phi)
        self._draw_axes(img, R)
        if len(edges) != 0:
            self._plot_edges(img, vertices, edges, R)

    def _draw_axes(self, img, R):
        axes_2d = np.dot(self.axes, R)
        axes_2d = axes_2d * self.scale + self.origin
        for axe in axes_2d:
            axe = axe.astype(int)
            cv2.line(img, tuple(axe[0]), tuple(axe[1]), (128, 128, 128), 1, cv2.LINE_AA)

    def _plot_edges(self, img, vertices, edges, R):
        vertices_2d = np.dot(vertices, R)
        edges_vertices = vertices_2d.reshape((-1, 2))[edges] * self.scale + self.origin
        for edge_vertices in edges_vertices:
            edge_vertices = edge_vertices.astype(int)
            cv2.line(img, tuple(edge_vertices[0]), tuple(edge_vertices[1]), (255, 255, 255), 1, cv2.LINE_AA)

    def _get_rotation(self, theta, phi):
        sin, cos = math.sin, math.cos
        return np.array([
            [ cos(theta),  sin(theta) * sin(phi)],
            [-sin(theta),  cos(theta) * sin(phi)],
            [ 0,                       -cos(phi)]
        ], dtype=np.float32)  # transposed

    @staticmethod
    def mouse_callback(event, x, y, flags, params):
        global previous_position, theta, phi, should_rotate, scale_dx, scale_dy
        if event == cv2.EVENT_LBUTTONDOWN:
            previous_position = [x, y]
            should_rotate = True
        if event == cv2.EVENT_MOUSEMOVE and should_rotate:
            theta += (x - previous_position[0]) / scale_dx * 6.2831  # 360 deg
            phi -= (y - previous_position[1]) / scale_dy * 6.2831 * 2  # 360 deg
            phi = max(min(3.1415 / 2, phi), -3.1415 / 2)
            previous_position = [x, y]
        if event == cv2.EVENT_LBUTTONUP:
            should_rotate = False


# read skeleton data from the csv 
def read_csv(filename):
    '''

    Parameters
    ----------
    filename : str
        Path to the CSV file.

    Returns
    -------
    df_new : dataframe
        Normalised coordinates of 3D pose.

    '''
    dataframe = pd.read_csv(filename, index_col='Body Part')
    # find the bbox of the player for crop
    xmax = -10000
    ymax = -10000
    zmax = -10000
    xmin = 10000
    ymin = 10000
    zmin = 10000
    
    # find the max/min value for each axis
    for key in dataframe.keys():
        data = list(dataframe[key][1:])
        data = list(map(float, data))
        
        data_num = np.array(data)
        data_num = data_num[np.where(~np.isnan(data_num))]
        
        keys = key.split('.')
        if len(keys) == 1:
            key_new = (keys[0], 'x')            
            xmax = max(xmax, np.max(data_num))
            xmin = min(xmin, np.min(data_num))
            
        elif len(keys) == 2 and keys[1] == '1':
            key_new = (keys[0], 'y')
            ymax = max(ymax, np.max(data_num))
            ymin = min(ymin, np.min(data_num))

        elif len(keys) == 2 and keys[1] == '2':
            key_new = (keys[0], 'y')
            zmax = max(zmax, np.max(data_num))
            zmin = min(zmin, np.min(data_num))
            
        if key == 'MidHip':
            data_midhip = data_num
        if key == 'Neck':
            data_neck = data_num
    
    # determine the center of x, y, z 
    xc = (np.mean(data_neck) + np.mean(data_midhip))/2
    yc = (ymax + ymin)/2   
    zc = (zmax + zmin)/2
    
    # determine the width, height and depth
    width = 2*max(xc-xmin, xmax-xc)
    height = (ymax - ymin)
    depth = (zmax - zmin)
    
    # select a cubic 
    sq = max(width, height)
    sq = max(sq, depth)
    sq = np.ceil(sq/100)*100
    depth = width = height = sq
    xmin = xc - sq/2
    ymin = yc - sq/2
    zmin = zc - sq/2
    
    # normalise the absolute coordinates with length of the cubic
    df = dict()
    for key in dataframe.keys():
        data = list(dataframe[key][1:])
        data = list(map(float, data))
        nan_idx = np.where(np.isnan(data))[0]
        if len(nan_idx) == len(data):
            data[:] = 0
        elif len(nan_idx) > 0:
            for jj in nan_idx:
                if jj == 0:
                    data[jj] = np.where(~np.isnan(data))[0][0]
                else:
                    data[jj] = data[jj-1]
                            
        keys = key.split('.')
        if len(keys) == 1:
            key_new = (keys[0], 'x')
            data = np.round(list((np.array(data)-xmin)/(width)), 5)
            
        elif len(keys) == 2 and keys[1] == '1':
            key_new = (keys[0], 'y')
            data = np.round(list((np.array(data)-ymin)/height), 5)
            
        elif len(keys) == 2 and keys[1] == '2':
            key_new = (keys[0], 'z')
            data = np.round(list((np.array(data)-zmin)/depth), 5)
        else:
            key_new = (keys[0], 'c')
            data = np.array(data)
        
        df[key_new] = data
    df_new = pd.DataFrame(df)
    return df_new


# create .mp4 file
def create_3d_video(output_path, df, parts=PARTS, skeleton=SKELETON_EDGES, output=True):
    '''

    Parameters
    ----------
    output_path : str
        Path for the created video.
    df : dataframe
        3D pose
    parts : list, optional
        The name of body parts. The default is PARTS.
    skeleton : narray, optional
        Indicating which two keypoints are connected. The default is SKELETON_EDGES.
    output : bool, optional
        True for storing the video file, otherwise only show the real-time window. The default is True.

    Returns
    -------
    None.

    '''
    # create a window for 3d visulisation
    canvas_3d = np.zeros((360, 640, 3), dtype=np.uint8)
    canvas_3d_window_name = 'Pose_3D'
    cv2.namedWindow(canvas_3d_window_name)
    cv2.setMouseCallback(canvas_3d_window_name, Plotter3d.mouse_callback)
    plotter = Plotter3d(canvas_3d.shape[:2],parts=parts, skeleton_edges=skeleton)
    depth = width = height = 200
    
    # if store the files
    if output:
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')   
        out = cv2.VideoWriter(output_path, fourcc, 25, (640, 360))
    
    # 3d skeleton coordinates in the window
    poses_3d_array = np.zeros((len(df), int(len(df.keys())/4), 3))
    for ii, part in enumerate(plotter.parts):
        poses_3d_array[:, ii, 0] = df[(part, 'x')]*width - 125
        poses_3d_array[:, ii, 1] = df[(part, 'y')]*height - 125
        poses_3d_array[:, ii, 2] = df[(part, 'z')]*depth
    
    # draw the skeleton
    for num in range(len(df)):
        poses_3d = poses_3d_array[num] 
        edges = plotter.skeleton_edges
        plotter.plot(canvas_3d, poses_3d, edges)
        if not output:
            cv2.imshow(canvas_3d_window_name, canvas_3d)
            delay = 1
            esc_code = 27
            p_code = 112
            space_code = 32    
            
            key = cv2.waitKey(delay)
            if key == esc_code:
                break
            if key == p_code:
                if delay == 1:
                    delay = 0
                else:
                    delay = 1
            if delay == 0:   # allow to rotate 3D canvas while on pause
                key = 0
                while (key != p_code
                        and key != esc_code
                        and key != space_code):
                    plotter.plot(canvas_3d, poses_3d, edges)
                    cv2.imshow(canvas_3d_window_name, canvas_3d)
                    key = cv2.waitKey(33)
                if key == esc_code:
                    break
                else:
                    delay = 1 
        else:
            out.write(canvas_3d)
    if output:
        out.release()


if __name__ == '__main__':
    # dataframe = read_csv(os.path.join(csv_dir, csv))
    dataframe = read_csv('../data/CSV_Solo_3d/AG_1a_Jaun.csv')
    df = dataframe
    create_3d_video('test.mp4', df)