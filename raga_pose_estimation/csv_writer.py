import os


def write_csv(person_dfs, output_dir, trial_no=None, flatten=False):
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
    for i, person_df in enumerate(person_dfs):

        if flatten:
            person_df.columns = [
                "_".join(col).strip() for col in person_df.columns.values
            ]

        # Output to CSV
        if trial_no:
            person_df.to_csv(
                os.path.join(output_dir, f"person{i}_trial_{trial_no}.csv"), float_format="%.3f"
            )
        else:
            person_df.to_csv(
                os.path.join(output_dir, f"person{i}.csv"), float_format="%.3f"
            )
