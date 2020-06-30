import os
import shutil
import sys

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
        create_model_video=True,
        create_overlay_video=True,
        body_parts=[
            OpenPoseParts.L_EYE,
            OpenPoseParts.NOSE,
            OpenPoseParts.R_EYE,
        ],
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
                    == "0,221.44,291.929,0.884024,218.301,306.0,0.946648,201.026,295.007,0.907743\n"
                )
            else:
                assert (
                    data[2]
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
            openpose_dir="notavalidpath",
            input_video="example_files/example_3people/short_video.mp4",
        )

    captured = capsys.readouterr()
    assert (
        captured.out
        == f"Detecting poses on example_files/example_3people/short_video.mp4.\nInvalid openpose path {invalid_path}.\n"
    )

    # Empty JSON path
    with pytest.raises(SystemExit):
        run_openpose.run_openpose(output_path, input_json=output_path)

    captured = capsys.readouterr()
    assert captured.out == f"No json files found in {output_path}.\n"


def test_openpose_cli(output_path):
    # Invalid cli command
    result = os.system(f"python run_openpose.py --output-dir {output_path}")
    assert result > 0

    # Valid cli command
    result = os.system(
        f"python run_openpose.py --input-json example_files/example_3people/output_json --output-dir {output_path} --upper-body-parts"
    )
    assert result == 0

    # Test CSVs have been created and the first lines are as we expect
    csv_files = ["person0.csv", "person1.csv"]
    for f in csv_files:
        filename = os.path.join(output_path, f)
        assert os.path.isfile(filename)
