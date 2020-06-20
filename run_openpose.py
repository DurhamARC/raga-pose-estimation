import os

import click
import cv2

from entimement_openpose.csv_writer import CSVWriter
from entimement_openpose.openpose_json_parser import OpenPoseJsonParser
from entimement_openpose.visualizer import Visualizer


@click.command()
@click.option('--input-video', default=None,
              help='Path to the video file on which to run openpose')
@click.option('--input-json', default=None,
              help='Path to a directory of previously generated openpose '
              'json files')
@click.option('--output-dir', prompt='Output directory',
              help='Path to the directory in which to output CSV files (and '
              'videos if required).')
@click.option('--openpose-dir',
              help='Path to the directory in which openpose is installed.')
@click.option('--create-model-video', is_flag=True, default=False,
              help='Whether to create a video showing the poses on a blank '
              'background')
@click.option('--create-overlay-video', is_flag=True, default=False,
              help='Whether to create a video showing the poses as an overlay')
@click.option('--width', default=0,
              help='Width of original video (mandatory for creating video if '
              ' not providing input-video)')
@click.option('--height', default=0,
              help='Height of original video (mandatory for creating video if '
              ' not providing input-video)')
def openpose_cli(output_dir, openpose_dir, input_video,
                 input_json, create_model_video,
                 create_overlay_video, width, height):
    """Runs openpose on the video, does post-processing, and outputs CSV
       files."""
    run_openpose(output_dir, openpose_dir, input_video,
                 input_json, create_model_video, create_overlay_video,
                 width, height)


def run_openpose(output_dir, openpose_dir=None, input_video=None,
                 input_json=None, create_model_video=False,
                 create_overlay_video=False, width=None, height=None):
    """Runs openpose on the video, does post-processing, and outputs CSV files.
    Non-click version to work from jupyter notebooks."""
    if not os.path.isdir(output_dir):
        print(f'No such directory {output_dir}.')
        exit(1)

    # Check input values
    if input_json is None and input_video is None:
        print("You must provide either an input video or input json files.")
        exit(1)

    if input_json is None and openpose_dir is None:
        print("You must provide a path to an openpose executable unless you "
              "provide input json.")
        exit(1)

    if input_video is None and create_overlay_video:
        print("You must provide an input video in order to create an overlay "
              "video.")
        exit(1)

    if input_video is None and create_model_video and \
       (width == 0 or height == 0):
        print("You must provide width and height of the original video in "
              "order to produce a model video.")
        exit(1)

    # Get full path to output directory
    output_dir = os.path.abspath(output_dir)

    # Run openpose if necessary
    if input_json:
        path_to_json = os.path.abspath(input_json)
    else:
        print(f'Detecting poses on {input_video}.')

        # Run openpose over the video
        openpose_dir = os.path.abspath(openpose_dir)

        # Calling out to the binary seems to be quicker than the python wrapper
        input_video = os.path.realpath(input_video)
        path_to_json = os.path.join(output_dir, 'json')
        os.system(f'cd {openpose_dir} && '
                  './build/examples/openpose/openpose.bin '
                  f'--video {input_video} '
                  f'--write_json {path_to_json} --display 0 --render-pose 0')

    # Get list of json files
    json_files = [pos_json for pos_json in os.listdir(path_to_json)
                  if pos_json.endswith('.json')]

    # Get array for dataframes
    body_keypoints_dfs = []

    # Loop through all json files in output directory
    # Each file is a frame in the video
    json_files.sort()
    print(len(json_files))
    for file in json_files:
        parser = OpenPoseJsonParser(os.path.join(path_to_json, file))
        body_keypoints_df = parser.get_multiple_keypoints([0, 1])
        body_keypoints_df.reset_index()
        body_keypoints_dfs.append(body_keypoints_df)

    print(len(body_keypoints_dfs))

    if create_model_video:
        print('Creating model video')

    if create_overlay_video:
        print('Creating overlay video')

    if create_model_video or create_overlay_video:
        if not width or not height:
            cap = cv2.VideoCapture(input_video)
            width = int(cap.get(3))
            height = int(cap.get(4))
            cap.release()

        visualizer = Visualizer(output_directory=output_dir)
        visualizer.create_videos_from_dataframes(
            'video',
            body_keypoints_dfs,
            width,
            height,
            create_blank=create_model_video,
            create_overlay=create_overlay_video,
            video_to_overlay=input_video
        )

    print(f'Saving CSVs to {output_dir}.')
    CSVWriter.writeCSV(body_keypoints_dfs, output_dir)


if __name__ == '__main__':
    openpose_cli()
