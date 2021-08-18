import numpy as np
import pandas as pd
import pytest

from entimement_openpose.openpose_parts import OpenPoseParts
from entimement_openpose.reshaper import reshape_dataframes


def _get_dummy_dataframes():
    data = [
        {
            "x": [2, 15],
            "y": [8, 5],
            "confidence": [0.1, 0.8],
        },
        {
            "x": [3, 10],
            "y": [5, 4],
            "confidence": [0.5, 0.9],
        },
        {
            "x": [4, np.nan],
            "y": [4, np.nan],
            "confidence": [0.7, np.nan],
        },
    ]
    dataframes = []
    for d in data:
        dataframes.append(
            pd.DataFrame(
                d,
                columns=["x", "y", "confidence"],
                index=pd.Index(
                    [
                        OpenPoseParts.R_EAR.value,
                        OpenPoseParts.R_EYE.value,
                    ]
                ),
            )
        )
    return dataframes


@pytest.fixture
def dummy_dataframes():
    return _get_dummy_dataframes()


@pytest.fixture
def single_frame_person_df():
    return reshape_dataframes([_get_dummy_dataframes()[0]])


@pytest.fixture
def three_frame_person_dfs():
    return reshape_dataframes(_get_dummy_dataframes())
