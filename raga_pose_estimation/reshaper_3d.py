import pandas as pd

# Jin modification
# add one dimension: (x, y, c) -> (x, y, z, c)


def reshape_dataframes(dataframes):
    """Combines the dataframes per frame into a dataframe per person
    across all frames. The returned dataframes contain columns for
    x, y, c for each body parts and rows representing each frame.

    If the dataframes contain multiple people, call
    sort_persons_by_x_position before this method

    Parameters
    ----------
    dataframes : list of DataFrame
        List of DataFrames as produced by OpenPoseJsonParser.
    """
    n_people = len(dataframes[0].columns) // 4
    all_person_dfs = []

    for i in range(n_people):
        person_dfs = [df.iloc[:, i * 4 : i * 4 + 4] for df in dataframes]
        variable_names = list(person_dfs[0].columns)

        all_dfs = pd.concat(person_dfs, keys=range(len(person_dfs)))

        # Unstack to get a row for each frame, with x, y, confidence at top level of cols,
        # and part at second level of cols
        all_dfs = all_dfs.unstack()
        all_dfs.columns.names = ["Variable", "Body Part"]

        # Reorder the column levels
        all_dfs = all_dfs.reorder_levels(["Body Part", "Variable"], axis=1)

        # Restack to group x, y, confidence for each body part
        all_dfs = all_dfs.stack(1).unstack()

        # Reindex and rename the variables so they're x, y, z, c
        all_dfs = all_dfs.reindex(variable_names, level="Variable", axis=1)
        all_dfs = all_dfs.rename(
            columns={
                variable_names[0]: "x",
                variable_names[1]: "y",
                variable_names[2]: "z",
                variable_names[3]: "c",
            },
            level="Variable",
        )

        all_person_dfs.append(all_dfs)

    return all_person_dfs
