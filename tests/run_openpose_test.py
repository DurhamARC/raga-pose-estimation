import os
import shutil
import sys

import pandas as pd
import pytest

sys.path.append("../")

from entimement_openpose.openpose_parts import OpenPoseParts
import run_openpose


@pytest.fixture
def output_path():
    # Gets path to output/tests/ and ensures it does not exist
    output_path = os.path.realpath("output/tests/")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    return output_path


def test_run_openpose(output_path):
    path_to_json = os.path.realpath("tests/test_json/")

    run_openpose.run_openpose(
        output_path,
        input_video="example_files/example_3people/short_video.mp4",
        input_json=path_to_json,
        number_of_people=2,
        create_model_video=True,
        create_overlay_video=True,
        body_parts=[
            OpenPoseParts.L_EYE,
            OpenPoseParts.NOSE,
            OpenPoseParts.R_EYE,
        ],
        smoothing_parameters=(21, 2),
    )

    # Check video files have been created and are about the size we expect
    video_files = ["video_overlay.avi", "video_blank.avi"]
    for f in video_files:
        filename = os.path.join(output_path, f)
        assert os.path.isfile(filename)
        size = os.path.getsize(filename)
        if f == video_files[0]:
            assert size > 1000000 and size < 10000000
        else:
            assert size > 100000 and size < 1000000

    # Test CSVs have been created and the first lines are as we expect
    csv_files = ["person0.csv", "person1.csv"]
    for f in csv_files:
        filename = os.path.join(output_path, f)
        assert os.path.isfile(filename)

        with open(filename) as fh:
            data = fh.readlines()
            assert len(data) == 28
            assert (
                data[0]
                == "Body Part,LEye,LEye,LEye,Nose,Nose,Nose,REye,REye,REye\n"
            )
            assert data[1] == "Variable,x,y,c,x,y,c,x,y,c\n"

            if f == csv_files[0]:
                assert (
                    data[2]
                    == "0,221.44146414455102,292.440453416149,0.8857431581027666,218.28011123658945,306.0235025409372,0.9471599440993789,199.7932987012986,295.00722473178996,0.9084977961603613\n"
                )
            else:
                assert (
                    data[2]
                    == "0,498.14364709203846,307.6110999435347,0.9068777549407112,485.10291360813085,324.6441072840203,0.8720559299830601,472.19120440429117,307.94809147374355,0.9198539649915302\n"
                )

        # Check we can read the CSV back into a DataFrame of the expected shape
        df = pd.read_csv(
            os.path.join(output_path, f), header=[0, 1], index_col=0
        )
        assert list(df.index.values) == list(range(26))
        assert df.columns.names == ["Body Part", "Variable"]
        assert df.shape == (26, 9)


def test_run_openpose_flatten(output_path):
    path_to_json = os.path.realpath("tests/test_json/")

    run_openpose.run_openpose(
        output_path,
        input_json=path_to_json,
        number_of_people=2,
        body_parts=[
            OpenPoseParts.L_EYE,
            OpenPoseParts.NOSE,
            OpenPoseParts.R_EYE,
        ],
        flatten=True,
    )

    # Test CSVs have been created and the first lines are as we expect
    csv_files = ["person0.csv", "person1.csv"]
    for f in csv_files:
        filename = os.path.join(output_path, f)
        assert os.path.isfile(filename)

        with open(filename) as fh:
            data = fh.readlines()
            assert len(data) == 27
            assert (
                data[0]
                == ",LEye_x,LEye_y,LEye_c,Nose_x,Nose_y,Nose_c,REye_x,REye_y,REye_c\n"
            )

            if f == csv_files[0]:
                assert (
                    data[1]
                    == "0,221.44,291.929,0.884024,218.301,306.0,0.946648,201.026,295.007,0.907743\n"
                )
            else:
                assert (
                    data[1]
                    == "0,495.607,307.671,0.911183,483.048,323.228,0.884607,470.521,307.588,0.932604\n"
                )


def test_run_openpose_invalid_output_path(capsys):
    # Create output directory
    output_path = os.path.realpath("output/tests/")
    os.makedirs(output_path, exist_ok=True)
    open(os.path.join(output_path, "invalid"), "x")

    with pytest.raises(SystemExit):
        run_openpose.run_openpose(output_path)

    captured = capsys.readouterr()
    assert (
        captured.out
        == f"Directory {output_path} exists and is not empty, so files would be overridden.\n"
    )


def test_run_openpose_invalid_parameters(capsys, output_path):
    # Run without sufficient parameters
    with pytest.raises(SystemExit):
        run_openpose.run_openpose(output_path)

    captured = capsys.readouterr()
    assert (
        captured.out
        == "You must provide either an input video or input json files.\n"
    )

    # Input video without json or openpose
    with pytest.raises(SystemExit):
        run_openpose.run_openpose(
            output_path,
            input_video="example_files/example_3people/short_video.mp4",
        )

    captured = capsys.readouterr()
    assert (
        captured.out
        == "You must provide a path to an openpose executable unless you provide input json.\n"
    )

    # Overlay without input video
    with pytest.raises(SystemExit):
        run_openpose.run_openpose(
            output_path,
            input_json="example_files/example_3people",
            create_overlay_video=True,
        )

    captured = capsys.readouterr()
    assert (
        captured.out
        == "You must provide an input video in order to create an overlay video.\n"
    )

    # Model video without dimensions or input video
    with pytest.raises(SystemExit):
        run_openpose.run_openpose(
            output_path,
            input_json="example_files/example_3people/output_json",
            create_model_video=True,
        )

    captured = capsys.readouterr()
    assert (
        captured.out
        == "You must provide width and height of the original video in order to produce a model video.\n"
    )

    # Invalid JSON path
    with pytest.raises(SystemExit):
        run_openpose.run_openpose(output_path, input_json="notavalidpath")

    captured = capsys.readouterr()
    invalid_path = os.path.realpath("notavalidpath")
    assert captured.out == f"Invalid input_json path {invalid_path}.\n"

    # Invalid openpose path
    with pytest.raises(SystemExit):
        run_openpose.run_openpose(
            output_path,
            openpose_dir=invalid_path,
            input_video="example_files/example_3people/short_video.mp4",
        )

    captured = capsys.readouterr()
    assert (
        captured.out
        == f"Detecting poses on example_files/example_3people/short_video.mp4...\n"
        f"Invalid openpose path {invalid_path}.\n"
    )

    # Invalid openpose path: path exists but does not contain openpose
    example_path = os.path.realpath("example_files")
    with pytest.raises(SystemExit):
        run_openpose.run_openpose(
            output_path,
            openpose_dir=example_path,
            input_video="example_files/example_3people/short_video.mp4",
        )

    captured = capsys.readouterr()
    assert (
        captured.out
        == f"Detecting poses on example_files/example_3people/short_video.mp4...\n"
        f"Unable to run openpose from {example_path}.\n"
    )

    # Empty JSON path
    with pytest.raises(SystemExit):
        run_openpose.run_openpose(output_path, input_json=output_path)

    captured = capsys.readouterr()
    assert (
        captured.out == f"Processing JSON from {output_path}...\n"
        f"No json files found in {output_path}.\n"
    )


def test_openpose_cli_part_group(output_path):
    # Valid cli command
    result = os.system(
        f"python run_openpose.py --input-json example_files/example_3people/output_json "
        f"--output-dir {output_path} --upper-body-parts -n 2"
    )
    assert result == 0

    # Test CSVs have been created and have expected headers
    csv_files = ["person0.csv", "person1.csv"]
    for f in csv_files:
        filename = os.path.join(output_path, f)
        assert os.path.isfile(filename)

        with open(filename) as fh:
            data = fh.readlines()
            assert len(data) == 127
            cols = data[0].split(",")
            assert len(cols) == 40
            assert cols[0] == "Body Part"
            cols = data[1].split(",")
            assert len(data[1].split(",")) == 40
            assert cols[0] == "Variable"


def test_openpose_cli_specified_parts(output_path):
    # Valid cli command with specific parts
    result = os.system(
        f"python run_openpose.py --input-json example_files/example_3people/output_json "
        f"--output-dir {output_path} --body-parts=LEye,REye,Nose -n 2"
    )
    assert result == 0

    # Test CSVs have been created and the first lines are as we expect
    csv_files = ["person0.csv", "person1.csv"]
    for f in csv_files:
        filename = os.path.join(output_path, f)
        assert os.path.isfile(filename)

        with open(filename) as fh:
            data = fh.readlines()
            assert len(data) == 127
            assert (
                data[0]
                == "Body Part,LEye,LEye,LEye,Nose,Nose,Nose,REye,REye,REye\n"
            )
            assert data[1] == "Variable,x,y,c,x,y,c,x,y,c\n"


def test_openpose_cli_invalid_args(output_path):
    # Invalid cli command
    result = os.system(f"python run_openpose.py --output-dir {output_path}")
    assert result > 0


def test_openpose_cli_invalid_parts(output_path):
    # Invalid cli command
    result = os.system(
        f"python run_openpose.py --input-json example_files/example_3people/output_json "
        f"--output-dir {output_path} --body-parts=Forehead,Chin"
    )
    assert result > 0
