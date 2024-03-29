# Raga Pose Estimation

<!-- TOC depthFrom:2 depthTo:2 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Introduction](#introduction)
- [Running the Google Colab script](#running-the-google-colab-script)
- [Installation of raga_pose_estimation Python library](#installation-of-ragaposeestimation-python-library)
- [Running the command line script](#running-the-command-line-script)
- [Post-processing options](#post-processing-options)
- [CSV format](#csv-format)
- [Other details](#other-details)
- [License](#license)

<!-- /TOC -->

## Introduction

![Video of a performer singing a Raga](https://github.com/durhamarc/raga-pose-estimation/blob/jo-branch/example_overlay_1.gif?raw=true)

Raga Pose Estimation comprises a set of tools to facilitate the use of pose estimation software for the analysis of human movement in music performance. It was developed with the study of Indian classical music in mind – hence the name – but can be applied to any other type of small musical ensemble. Raga Pose Estimation was developed by staff in the Music Department and Advanced Research Computing at Durham University, as part of the EU Horizon 2020 FET project EnTimeMent – Entrainment & synchronization at multiple TIME scales in the MENTal foundations of expressive gesture. 

 

The code, which is available as a library and Colab script, utilizes the [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) tool to facilitate the analysis of musical performance. The Colab script does this by enabling the user to carry out pose estimation online, avoiding the need for a suitable local GPU. Post-processing functions turn the large numbers of JSON files (one per video frame) output by OpenPose into single CSV files per performer. Cropping and trimming of videos, applying confidence levels and smoothing functions to the output, are amongst the other functions that are built into a single process. 

We include some [example_files](https://github.com/DurhamARC/raga-pose-estimation/tree/jo-branch/example_files) to use with the system. Other suitable video examples can be found on Open Science Framework, see for example [North Indian Raga Performance](https://osf.io/nkjgz/). OpenPose proves effective for the extraction of movement information from static videos of music performance, although its performance can suffer in cases of poor lighting, occlusion (body parts are obscured by other objects), when limbs of different individuals overlap, or when tracked body parts leave the video frame altogether. The system is not dependent on any particular video resolution, aspect ratio or frame rate. 

This Readme describes how to use the scripts, whether on Colab or by installing the library locally. Even if you use the Colab script, the text below on use of the library may come in useful as it contains some more detailed information about the functions.  


![Video of three performers](https://github.com/durhamarc/raga-pose-estimation/blob/jo-branch/example_overlay_3.gif?raw=true)

## Running the Google Colab script
Open [RagaPoseEstimationColab.ipynb](RagaPoseEstimationColab.ipynb) and click 'Run in Colab'.


## 1. Set up <br />
Run the first four cells in the Colab. This will take approximately half an hour to run.<br />
![A video showing how to run a Colab cell](https://github.com/durhamarc/raga-pose-estimation/blob/jo-branch/read_me_images/1click.gif?raw=true) 

## 2. Input Parameters <br />
Once the first cells have run you will see this form:<br />
![A picture showing the layout of the parameters form](https://github.com/durhamarc/raga-pose-estimation/blob/jo-branch/read_me_images/params.png?raw=true)<br />

The inputs required depend on what type of files you are using. <br />

### Note: files can be uploaded and downloaded to Google Drive and accessed through the files tab

### 2a. With a Video <br />

### Additional OpenPose Arguements
There are additional arguments you can pass to OpenPose. See the [OpenPose documentation](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/flags.hpp) for a full list of options. 

#### Input a video URL <br />
![A picture showing where to input video URL](https://github.com/durhamarc/raga-pose-estimation/blob/jo-branch/read_me_images/input-video.png?raw=true)<br />

#### OPTIONAL: Choose the cropping parameters<br />
(Performance of OpenPose can sometimes be improved by cropping a video to include only the person or persons of interest. Cropping can be carried out as part of the Colab script by entering the parameters here.) 
![A video showing the input of the cropping parameters](https://github.com/durhamarc/raga-pose-estimation/blob/jo-branch/read_me_images/3crop.gif?raw=true)<br />

#### Choose the number of people in the video<br />
![A video showing the sliding number of people parameters](https://github.com/durhamarc/raga-pose-estimation/blob/jo-branch/read_me_images/4numberpeople.gif?raw=true)<br />

#### Type the names of the performers<br />
Type their names with commas inbetween<br />
<performer names>

#### Choose the video outputs desired<br />
You can create a model video (just showing the OpenPose skeleton), and/or an overlay video (the skeleton from the model video added to the input video). We recommend writing the output directly to a Google Drive folder: if you choose not to do this, you can click the ‘Download results’ box here. <br />
![A video showing how to choose the output videos](https://github.com/durhamarc/raga-pose-estimation/blob/jo-branch/read_me_images/6output.gif?raw=true)<br />

### 2b. With a JSON OpenPose Output<br />
If you have already run the system and want to re-run the post-processing with different parameters, you can start with the JSON folder  
from the previous run. Input its url or path here. <br />
![A picture showing where to input json URL](https://github.com/durhamarc/raga-pose-estimation/blob/jo-branch/read_me_images/input-json.png?raw=true)<br />

### 2c. With a Multiple JSON Outputs<br />
Input a folder containing folders in the same structure as [example_files](https://github.com/DurhamARC/raga-pose-estimation/tree/jo-branch/example_files) with videos if output videos desired.<br />
![A picture showing where to input batch files folder URL](https://github.com/durhamarc/raga-pose-estimation/blob/jo-branch/read_me_images/input-batch.png?raw=true)<br />

## 3. Choose Confidence Threshold<br />
OpenPose outputs a confidence level for each estimate of the x- and y- position of a body part. It may sometimes be useful to set a confidence threshold to eliminate some poorly estimated data (which can then be interpolated). Trial and error is necessary to establich a suitable confidence level. (Set to 0 by default.)<br />
![A picture showing how to put in confidence](https://github.com/durhamarc/raga-pose-estimation/blob/jo-branch/read_me_images/confidence.png?raw=true)<br />

## 4. Choose Smoothing Parameter<br />
To reduce jitter in the model, a smoothing function (Savitzky-Golay filter) can be applied. You can choose the Smoothing window, which determines the number of previous frames used in the smoothing calculation, and the Smoothing polynomial order, which is the order of the polynomial used in the fitting function. The smoother is set to off by default. <br />

![A video showing how to use the smoothing parameters](https://github.com/durhamarc/raga-pose-estimation/blob/jo-branch/read_me_images/7smoothing.gif?raw=true)<br />

## 5. Choose the Body Parts<br />
You can select which body parts to detect by choosing from options such as all body parts, upper body parts only (which is best for Indian classical music), lower body parts only, or choose specific body parts.<br />
![A video showing how to choose the output videos](https://github.com/durhamarc/raga-pose-estimation/blob/jo-branch/read_me_images/bodyparts.png?raw=true)<br />

## 6. Choose CSV Format<br />
Select whether you want the CSV data in the output to be flattened (a flattened CSV file includes only one header row, while the unflattened file includes two header rows).<br />
![A picture showing the box to select for CSV flattening](https://github.com/durhamarc/raga-pose-estimation/blob/jo-branch/read_me_images/csvflattening.png?raw=true)<br />

## 7. Choose Name of Trial<br />
Enter a name for the trial, which will be included in the file names.<br />
![A picture showing the box to select for CSV flattening](https://github.com/durhamarc/raga-pose-estimation/blob/jo-branch/read_me_images/trial.png?raw=true)<br />

## 8. Click Generate Parameters<br />
![A video showing how to generate parameters](https://github.com/durhamarc/raga-pose-estimation/blob/jo-branch/read_me_images/8generate.gif?raw=true)<br />

## 9. Run Next Cell<br />
This cell will execute the OpenPose/post-processing process using the specified parameters. The duration of the process will depend on the size of the video and may take anywhere from 5 minutes to several hours to complete.<br />
![A picture showing the cell to run the processing](https://github.com/durhamarc/raga-pose-estimation/blob/jo-branch/read_me_images/run.png?raw=true)<br />

## Installation of raga_pose_estimation Python library

### Requirements

  - pandas=1.3.5  
  - numpy
  - scipy
  - python=3.7
  - opencv
  - click
  - pytest
  - pytest-cov
  - coverage[toml]
  - black
  - pre-commit
  - ffmpeg-python
  - pymediainfo=6.0.1

### How to install dependencies

Install dependencies of `raga_pose_estimation` using [conda](https://docs.conda.io/projects/conda/en/latest/index.html) or [miniconda](https://docs.conda.io/en/latest/miniconda.html):

```
conda env create -f environment.yml
```

Alternatively use `pip` to install the packages listed in [`environment.yml`](environment.yml).

## Running the command line script

```
$ python run_pose_estimation.py --help
Usage: run_pose_estimation.py [OPTIONS]

  Runs openpose on the video, does post-processing, and outputs CSV files.
  See cli docs for parameter details.

Options:
  -v, --input-video TEXT          Path to the video file on which to run
                                  openpose

  -j, --input-json TEXT           Path to a directory of previously generated
                                  openpose json files

  -bf, --batch-folder TEXT        Path to directory of folders with subfolders
                                  with JSON files and optional videos

  -o, --output-dir TEXT           Path to the directory in which to output CSV
                                  files (and videos if required).

  -r, --crop-rectangle INTEGER...
                                  Coordinates of rectangle to crop the video
                                  before processing, in the form w h x y where
                                  w and h are the width and height of the
                                  cropped rectangle and (x,y) is the top-left
                                  of the rectangle, as measured in pixels from
                                  the top-left corner.

  -n, --number-of-people INTEGER  Number of people to include in output.
  -O, --openpose-dir TEXT         Path to the directory in which openpose is
                                  installed.

  -a, --openpose-args TEXT        Additional arguments to pass to OpenPose.
                                  See https://github.com/CMU-Perceptual-
                                  Computing-Lab/openpose/blob/master/include/o
                                  penpose/flags.hpp for a full list of
                                  options.

  -m, --create-model-video        Whether to create a video showing the poses
                                  on a blank background

  -V, --create-overlay-video      Whether to create a video showing the poses
                                  as an oVerlay

  -w, --width INTEGER             Width of original video (mandatory for
                                  creating video if  not providing input-
                                  video)

  -h, --height INTEGER            Height of original video (mandatory for
                                  creating video if  not providing input-
                                  video)

  -c, --confidence-threshold FLOAT
                                  Confidence threshold. Items with a
                                  confidence lower than the threshold will be
                                  replaced by values from a previous frame.

  -s, --smoothing-parameters <INTEGER INTEGER>...
                                  Window and polynomial order for smoother.
                                  See README for details.

  -b, --body-parts TEXT           Body parts to include in output. Should be a
                                  comma-separated list of strings as in the
                                  list at https://github.com/CMU-Perceptual-
                                  Computing-Lab/openpose/blob/master/doc/outpu
                                  t.md#keypoint-ordering-in-cpython, e.g.
                                  "LEye,RElbow". Overrides --upper-body-parts
                                  and --lower-body-parts.

  -u, --upper-body-parts          Output upper body parts only
  -l, --lower-body-parts          Output lower body parts only
  -f, --flatten TEXT              Export CSV in flattened format, i.e. with a
                                  single header row (see README)
  -tn, --trial_number INTEGER     Trial number of run.
  -p, --performer_names"          Names of performers from left to right.

  --help                          Show this message and exit.
```

### Examples

Run OpenPose on a video to produce output CSVs (1 per person) in the `output` directory
and an overlay video, cropping the video to width 720, height 800 from (600, 50):

```bash
python run_pose_estimation.py --input-video=example_files/example_1person/short_video.mp4 --openpose-dir=../openpose --output-dir=output --create-overlay-video --crop-rectangle 720 800 600 50
```

Parse existing JSON files created by OpenPose to produce 1 CSV per person in the `output` folder, showing only upper body parts, outputting up to 3 people, and using the confidence_threshold and smoothing to improve the output (using short form of arguments):

```bash
python run_pose_estimation.py -j example_files/example_3people/output_json -o output -u -n 3 -s 21 2 -c 0.7
```

### Quick run

The script `run_samples.sh` runs a sensible set of default options on the two examples in example_files, producing both overlay and model videos.

```bash
./run_samples.sh
```

## Post-processing options

There are two ways to reduce the jitter from the OpenPose output: using a confidence threshold and using the smoother.

### Confidence Threshold

Applying a confidence threshold removes values with a confidence level below the threshold and replaces them with the value from the previous frame.

### Smoothing

Setting the smoothing parameters applies a smoothing function to reduce the jitter between frames. The parameters are:

 * **smoothing window**: the length of the smoothing window, which needs to be odd. The higher it is, the more frames are taken into account for smoothing; between 11 and 33 or so seems to work well. If it's too big it will affect/delay genuine movements.

 * **polyorder**: the order of the polynomial used in the fitting function. It needs to be smaller than the smoothing window (2 seems to work well, 1 connects with straight lines, etc.)


## CSV format

The CSVs which are output give the positions for a single person. They contain one line per frame, and columns showing the x, y and confidence for each body part.

With `flatten=False` (default), the CSV output looks like this (note that this is a minimal example showing only the eyes):

```csv
Body Part,LEye,LEye,LEye,REye,REye,REye
Variable,x,y,c,x,y,c
0,977.627,285.955,0.907646,910.046,271.252,0.92357
1,974.83,286.009,0.910094,909.925,271.277,0.925763
...
```

which in table form looks like this:

| Body Part | LEye     | LEye     | LEye      | REye     | REye     | REye      |
|-----------|----------|----------|-----------|----------|----------|-----------|
| **Variable**  | **x**        | **y**        | **c**         | **x**        | **y**        | **c**         |
| 0         | 977\.627 | 285\.955 | 0\.907646 | 910\.046 | 271\.252 | 0\.92357  |
| 1         | 974\.83  | 286\.009 | 0\.910094 | 909\.925 | 271\.277 | 0\.925763 |


If you export CSVs with `flatten=False`, they can be read back into a `pandas.DataFrame` as follows:

```python
df = pd.read_csv(csv_path, header=[0,1], index_col=0)
```

The dataframe will have `MultiIndex` columns to allow easier access to the values.

With `flatten=True`, the CSV output looks like this:

```csv
,LEye_x,LEye_y,LEye_c,REye_x,REye_y,REye_c
0,977.627,285.955,0.907646,910.046,271.252,0.92357
1,974.83,286.009,0.910094,909.925,271.277,0.925763
...
```

which in table form looks like:

|   | LEye\_x  | LEye\_y  | LEye\_c   | REye\_x  | REye\_y  | REye\_c   |
|---|----------|----------|-----------|----------|----------|-----------|
| 0 | 977\.627 | 285\.955 | 0\.907646 | 910\.046 | 271\.252 | 0\.92357  |
| 1 | 974\.83  | 286\.009 | 0\.910094 | 909\.925 | 271\.277 | 0\.925763 |

## Other details
The files with suffix like '_3d' and '_adaptive' correspond to the process of specific pose data.
The folder 'utils' includes some useful tools to process the data. Please find more details from 'utils/README.md'.§


## License

The code in this repository is licensed under the [MIT License](LICENSE).

Note that in order to run OpenPose you must read and accept the
[OpenPose License](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE).
OpenPose is free for non-commercial use only; see the
[OpenPose README](https://github.com/CMU-Perceptual-Computing-Lab/openpose#license)
for further details.

