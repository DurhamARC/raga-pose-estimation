import os
import glob
import click
import cv2
from sys import exit

from raga_pose_estimation.csv_writer import write_csv
from raga_pose_estimation.openpose_json_parser import OpenPoseJsonParser
from raga_pose_estimation.openpose_parts import (
    OpenPosePartGroups,
    OpenPoseParts,
)
from raga_pose_estimation.reshaper import reshape_dataframes
from raga_pose_estimation.smoother import Smoother
from raga_pose_estimation.video_utils import crop_video
from raga_pose_estimation.visualizer import Visualizer
from raga_pose_estimation.audio_combiner import Audio


@click.command()
@click.option(
    "-v",
    "--input-video",
    default=None,
    help="Path to the video file on which to run openpose",
)
@click.option(
    "-j",
    "--input-json",
    default=None,
    help="Path to a directory of previously generated openpose json files",
)
@click.option(
    "-bf",
    "--batch-folder",
    default=None,
    help="Path to a directory of folders for batch processing",
)
@click.option(
    "-o",
    "--output-dir",
    prompt="Output directory",
    help="Path to the directory in which to output CSV files (and "
    "videos if required).",
)
@click.option(
    "-r",
    "--crop-rectangle",
    nargs=4,
    type=int,
    default=None,
    help="Coordinates of rectangle to crop the video before processing, in "
    "the form w h x y where w and h are the width and height of the cropped "
    "rectangle and (x,y) is the top-left of the rectangle, as measured in "
    "pixels from the top-left corner.",
)
@click.option(
    "-n",
    "--number-of-people",
    default=1,
    help="Number of people to include in output.",
)
@click.option(
    "-O",
    "--openpose-dir",
    help="Path to the directory in which openpose is installed.",
)
@click.option(
    "-a",
    "--openpose-args",
    help="Additional arguments to pass to OpenPose. See "
    "https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/flags.hpp"
    " for a full list of options.",
)
@click.option(
    "-m",
    "--create-model-video",
    is_flag=True,
    default=False,
    help="Whether to create a video showing the poses on a blank "
    "background",
)
@click.option(
    "-V",
    "--create-overlay-video",
    is_flag=True,
    default=False,
    help="Whether to create a video showing the poses as an oVerlay",
)
@click.option(
    "-w",
    "--width",
    default=0,
    help="Width of original video (mandatory for creating video if "
    " not providing input-video)",
)
@click.option(
    "-h",
    "--height",
    default=0,
    help="Height of original video (mandatory for creating video if "
    " not providing input-video)",
)
@click.option(
    "-c",
    "--confidence-threshold",
    default=0.0,
    help="Confidence threshold. Items with a confidence lower than "
    "the threshold will be replaced by values from a previous frame.",
)
@click.option(
    "-s",
    "--smoothing-parameters",
    default=(None, None),
    type=(int, int),
    help="Window and polynomial order for smoother. See README for "
    "details.",
)
@click.option(
    "-b",
    "--body-parts",
    default=None,
    help="Body parts to include in output. Should be a "
    "comma-separated list of strings as in the list at "
    "https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#keypoint-ordering-in-cpython"
    ', e.g. "LEye,RElbow". Overrides --upper-body-parts and '
    "--lower-body-parts.",
)
@click.option(
    "-u",
    "--upper-body-parts",
    "bodypartsgroup",
    flag_value="upper",
    help="Output upper body parts only",
)
@click.option(
    "-l",
    "--lower-body-parts",
    "bodypartsgroup",
    flag_value="lower",
    help="Output lower body parts only",
)
@click.option(
    "-f",
    "--flatten",
    "flatten",
    default=False,
    help="Export CSV in flattened format, i.e. with a single header row (see README)",
)

@click.option(
    "-tn",
    "--trial_name",
    type=str,
    default=None,
    help="Trial name for this run",
)

@click.option(
    "-p",
    "--performer_names",
    default=range(1,10),
    help="Performer's names from left to right",
    multiple=True
)

def openpose_cli(
    output_dir,
    openpose_dir,
    openpose_args,
    input_video,
    input_json,
    batch_folder,
    crop_rectangle,
    number_of_people,
    create_model_video,
    create_overlay_video,
    width,
    height,
    confidence_threshold,
    smoothing_parameters,
    body_parts,
    bodypartsgroup,
    flatten,
    trial_name,
    performer_names
):
    """Runs openpose on the video, does post-processing, and outputs CSV
    files. See cli docs for parameter details."""
    body_parts_list = None
    if body_parts:
        try:
            body_parts_list = [
                OpenPoseParts(part) for part in body_parts.split(",")
            ]
        except ValueError as e:
            click.echo(f"Invalid body-parts value {body_parts}: {e}")
            exit(1)
    elif bodypartsgroup == "upper":
        body_parts_list = OpenPosePartGroups.UPPER_BODY_PARTS
    elif bodypartsgroup == "lower":
        body_parts_list = OpenPosePartGroups.LOWER_BODY_PARTS

    if smoothing_parameters == (None, None):
        smoothing_parameters = None

    if batch_folder != None:
        multiple_videos(output_dir,
            openpose_dir,
            openpose_args,
            batch_folder,
            crop_rectangle,
            number_of_people,
            create_model_video,
            create_overlay_video,
            width,
            height,
            confidence_threshold,
            smoothing_parameters,
            body_parts_list,
            flatten,
            performer_names)
    else:
        run_pose_estimation(
                output_dir,
                openpose_dir,
                openpose_args,
                input_video,
                input_json,
                crop_rectangle,
                number_of_people,
                create_model_video,
                create_overlay_video,
                width,
                height,
                confidence_threshold,
                smoothing_parameters,
                body_parts_list,
                flatten,
                trial_name,
                performer_names
            )



def run_pose_estimation(
    output_dir,
    openpose_dir=None,
    openpose_args=None,
    input_video=None,
    input_json=None,
    crop_rectangle=None,
    number_of_people=1,
    create_model_video=False,
    create_overlay_video=False,
    width=0,
    height=0,
    confidence_threshold=0,
    smoothing_parameters=None,
    body_parts=None,
    flatten=False,
    trial_name=None,
    performer_names=None
):
    """Runs openpose on the video, does post-processing, and outputs CSV files.
    Non-click version to work from jupyter notebooks.

    Parameters
    ----------
    output_dir : str
        Path to the directory in which to output CSV
        files (and videos if required).
    openpose_dir : str
        Path to the directory in which openpose is
        installed.
    openpose_args : str
        Additional arguments to pass to openpose.
    input_video : str
        Path to the video file on which to run
        openpose.
    crop_rectangle : tuple(int)
        Coordinates for cropping the video before
        processing. Should be a tuple (width, height, x, y)
        giving the width and height of the cropped
        rectangle and the coordinates (x,y) of the
        top-left of the rectangle, as measured in
        pixels from the top-left corner.
    input_json : str
        Path to a directory of previously generated
        openpose json files.
    number_of_people : int
        Number of people for which to output results.
    create_model_video : bool
        Whether to create a video showing the poses on
        a blank background.
    create_overlay_video : type
        Whether to create a video showing the poses as
        an overlay.
    width : int
        Width of original video (mandatory for
        creating video if not providing input_video).
    height : int
        Height of original video (mandatory for
        creating video if not providing input_video).
    confidence_threshold : float
        Confidence threshold. Items with a confidence
        lower than the threshold will be replaced by
        values from a previous frame.
    smoothing_parameters : (int, int)
        Pair of parameters (smoothing_window, polyorder)
        defining the smoothing function applied. If None,
        no smoothing is attempted. See README.
    body_parts : list of OpenPoseParts
        Body parts to include in output.
    flatten : type
        Export CSV in flattened format, i.e. with a
        single header row (see README).
    """
    # Check output directory
    output_dir = output_dir + str(trial_name)
    output_dir = os.path.abspath(output_dir)
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        print(
            f"Directory {output_dir} exists and is not empty, so files would be overridden."
        )
        exit(1)

    # Check input values
    if input_json is None and input_video is None:
        print("You must provide either an input video or input json files.")
        exit(1)

    if input_json is None and openpose_dir is None:
        print(
            "You must provide a path to an openpose executable unless you "
            "provide input json."
        )
        exit(1)

    if input_video is None and crop_rectangle:
        print("You must provide an input video in order to crop the video.")
        exit(1)

    if input_video is None and create_overlay_video and batch_folder != None:
        print(
            "You must provide an input video in order to create an overlay "
            "video."
        )
        exit(1)

    if (
        input_video is None
        and create_model_video
        and (width == 0 or height == 0)
    ):
        print(
            "You must provide width and height of the original video in "
            "order to produce a model video."
        )
        exit(1)

    if crop_rectangle and (
        type(crop_rectangle) != tuple
        or len([c for c in crop_rectangle if type(c) == int]) != 4
    ):
        print("You must provide 4 integer coordinates to crop_rectangle.")
        exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Run openpose if necessary
    if input_json:
        path_to_json = os.path.abspath(input_json)
        if not os.path.exists(path_to_json):
            print(f"Invalid input_json path {path_to_json}.")
            exit(1)
        else:
            print(f"Processing JSON from {path_to_json}...")
    else:
        input_video = os.path.realpath(input_video)

        # Crop the video before processing
        if crop_rectangle:
            print(f"Cropping video {input_video}...")
            try:
                input_video = crop_video(
                    input_video, output_dir, *crop_rectangle
                )
                print(f"Cropped video at {input_video}...")
            except Exception as e:
                print(
                    f"Unable to crop video {input_video} with coords {crop_rectangle}. Error was: {e}"
                )
                exit(1)

        print(f"Detecting poses on {input_video}...")

        # Run openpose over the video
        openpose_dir = os.path.abspath(openpose_dir)
        if not os.path.exists(openpose_dir):
            print(f"Invalid openpose path {openpose_dir}.")
            exit(1)

        # Calling out to the binary seems to be quicker than the python wrapper
        path_to_json = os.path.join(output_dir, "json")
        cmd = (
            f"cd {openpose_dir} && "
            "./build/examples/openpose/openpose.bin "
            f"--video {input_video} "
            f"--write_json {path_to_json} --display 0 --render-pose 0"
        )
        if openpose_args:
            cmd = f"{cmd} {openpose_args}"

        try:
            # If running in Colab, need to use ipython's system call
            ip = get_ipython()
            result = ip.system_piped(cmd)
        except NameError:
            # Otherwise (if get_ipython doesn't exist) we can just use os.system
            result = os.system(cmd)

        if result:
            print(f"Unable to run openpose from {openpose_dir}.")
            exit(1)

    # Get list of json files
    json_files = [
        pos_json
        for pos_json in os.listdir(path_to_json)
        if pos_json.endswith(".json")
    ]

    if len(json_files) != 0:
        print(f"JSON folder in {path_to_json}.")

        # Get array for dataframes
        body_keypoints_dfs = []

        # Loop through all json files in output directory
        # Each file is a frame in the video
        json_files.sort()
        previous_body_keypoints_df = None

        for file in json_files:
            parser = OpenPoseJsonParser(os.path.join(path_to_json, file))
            body_keypoints_df = parser.get_multiple_keypoints(
                list(range(number_of_people)),
                body_parts,
                confidence_threshold,
                previous_body_keypoints_df,
            )
            body_keypoints_df = parser.sort_persons_by_x_position(
                body_keypoints_df
            )
            body_keypoints_df.reset_index()
            body_keypoints_dfs.append(body_keypoints_df)

            
            people_found = int(body_keypoints_df.shape[1]/3)
            if number_of_people == people_found:

                previous_body_keypoints_df = body_keypoints_df

        person_dfs = reshape_dataframes(body_keypoints_dfs)

    if smoothing_parameters:
        print("Smoothing output...")
        smoother = Smoother(*smoothing_parameters)
        smoothed_person_dfs = smoother.smooth(person_dfs)

    if create_model_video or create_overlay_video:
        if not width or not height:
            cap = cv2.VideoCapture(input_video)
            width = int(cap.get(3))
            height = int(cap.get(4))
            cap.release()

        visualizer = Visualizer(output_directory=output_dir)
        audio = Audio(input_video_path = input_video, output_directory=output_dir)

        if create_model_video:
            print("Creating model video...")
            visualizer.create_video_from_dataframes(
                "video", person_dfs, width, height, input_video_path=input_video
            )
            

        if create_overlay_video:
            print("Creating overlay video...")
            visualizer.create_video_from_dataframes(
                "video",
                person_dfs,
                width,
                height,
                input_video_path=input_video,
                create_overlay=create_overlay_video,
                video_to_overlay=input_video,
            )

            print("Adding audio...")
            try:
                audio.audio_combiner(input_video_path=input_video, output_directory = output_dir)
            except Exception:
                pass

    print(f"Saving CSVs to {output_dir}...")
    write_csv(person_dfs, output_dir, trial_name, performer_names, flatten=flatten, smoothed = False)
    if smoothed_person_dfs:
        write_csv(smoothed_person_dfs, output_dir, trial_name, performer_names, flatten=flatten, smoothed = True)
    print("Done.")

def multiple_videos(output_dir,
                openpose_dir,
                openpose_args,
                batch_folder,
                crop_rectangle,
                number_of_people,
                create_model_video,
                create_overlay_video,
                width,
                height,
                confidence_threshold,
                smoothing_parameters,
                body_parts_list,
                flatten,
                performer_names):
    contents = os.listdir(batch_folder)
    # If single run
    if not any(".json" in file for file in contents):
        contents = [x for x in contents if not x.startswith('.')]
        for input_folder in contents:
            multi_input_json = batch_folder + input_folder + "/output_json/"
            multi_trial_name = input_folder
            multi_output_dir = multi_name(output_dir,
                confidence_threshold,
                smoothing_parameters)
            multi_input_video = find_input_video(batch_folder, input_folder)

            run_pose_estimation(
                multi_output_dir,
                openpose_dir,
                openpose_args,
                multi_input_video,
                multi_input_json,
                crop_rectangle,
                number_of_people,
                create_model_video,
                create_overlay_video,
                width,
                height,
                confidence_threshold,
                smoothing_parameters,
                body_parts_list,
                flatten,
                multi_trial_name,
                performer_names
            )

def multi_name(output_dir,
                confidence_threshold,
                smoothing_parameters):
    con = str(confidence_threshold)[0:3]
    smo = str(smoothing_parameters)[1:-1].replace(', ', '_')
    return f'{output_dir}_c{con}_s{smo}'

def find_input_video(batch_folder, input_folder):
    print(batch_folder)
    video_path = batch_folder + input_folder + '/*.mp4'
    mp4_files = glob.glob(video_path)
    print(mp4_files)
    if len(mp4_files) == 0:
        return None 
    if len(mp4_files) == 1:
        print("Input video found.")
        return mp4_files[0]
    else:
        print("Too many video files in folder.")
        return None 

if __name__ == "__main__":
    openpose_cli()
