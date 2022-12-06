import os


def write_csv(person_dfs, output_dir, trial_no=None, performers_names= None, flatten=False):
    """Creates CSVs in the given output directory. Each CSV contains
    details for 1 person, with columns for x, y, c for each body parts and
    rows representing each frame.

    Parameters
    ----------
    person_dfs : list of DataFrame
        List of DataFrames as produced by reshape_dataframes.
    output_dir : str
        Directory in which to output CSV files.
    flatten : bool
        Whether to flatten the CSV multi-line headers to a single row
        (see README)
    """
    performer_dict = performer_to_dict(performers_names, person_dfs)

    trial_no = trial_number(trial_no)

    for i, person_df in enumerate(person_dfs):

        if flatten:
            person_df.columns = [
                "_".join(col).strip() for col in person_df.columns.values
            ]

        # Output to CSV
        person_df.to_csv(
            os.path.join(output_dir, f"{performer_dict[i]}{trial_no}.csv"), float_format="%.3f"
        )


def performer_to_dict(performers_names, person_dfs):
    performer_dict = {}
    if performers_names:
        for i, name in enumerate(performers_names):
            performer_dict[i] = name
    else:
        for i, name in enumerate(person_dfs):
            performer_dict[i] = f"person_{i}"
    return performer_dict


def trial_number(trial_no):
    if trial_no:
        trial_no = trial_no
    else:
        trial_no = ""
    return trial_no