# OpenPose for EnTimeMent

Library and CoLab script to facilitate running [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and
to do some post-processing.

This library was created for the [EnTimeMent](https://entimement.dibris.unige.it) project.

<!-- TOC depthFrom:2 depthTo:2 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Running the CoLab script](#running-the-colab-script)
- [Installation of `entimement_openpose` python library](#installation-of-entimementopenpose-python-library)
- [Running the command line script](#running-the-command-line-script)
- [Post-processing options](#post-processing-options)
- [CSV format](#csv-format)

<!-- /TOC -->

## Running the CoLab script
Open [OpenPose_Colab.ipynb](OpenPose_Colab.ipynb) and click 'Run in CoLab'.

## Installation of `entimement_openpose` python library
Install dependencies of entimement_openpose using [conda](https://docs.conda.io/projects/conda/en/latest/index.html) or [miniconda](https://docs.conda.io/en/latest/miniconda.html):

```
conda env create -f environment.yml
```

Alternatively use `pip` to install the packages listed in [`environment.yml`](environment.yml).

## Running the command line script

```
$ python run_openpose.py --help
Usage: run_openpose.py [OPTIONS]

  Runs openpose on the video, does post-processing, and outputs CSV files.
  See cli docs for parameter details.

Options:
  -v, --input-video TEXT          Path to the video file on which to run
                                  openpose

  -j, --input-json TEXT           Path to a directory of previously generated
                                  openpose json files

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

  --help                          Show this message and exit.
```

### Examples

Run OpenPose on a video to produce output CSVs (1 per person) in the `output` directory
and an overlay video, cropping the video to width 720, height 800 from (600, 50):

```bash
python run_openpose.py --input-video=example_files/example_1person/short_video.mp4 --openpose-dir=../openpose --output-dir=output --create-overlay-video --crop-rectangle 720 800 600 50
```

Parse existing JSON files created by OpenPose to produce 1 CSV per person in the `output` folder, showing only upper body parts, outputting up to 3 people, and using the confidence_threshold and smoothing to improve the output (using short form of arguments):

```bash
python run_openpose.py -j example_files/example_3people/output_json -o output -u -n 3 -s 21 2 -c 0.7
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
