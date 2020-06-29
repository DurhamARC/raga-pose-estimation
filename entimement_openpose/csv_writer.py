import os
import pandas as pd


class CSVWriter:
    def writeCSV(dataframes, output_dir):
        n_people = n_people = len(dataframes[0].columns) // 3

        for i in range(n_people):
            person_dfs = [df.iloc[:, i * 3 : i * 3 + 3] for df in dataframes]
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

            # Reindex and rename the variables so they're x, y, c
            all_dfs = all_dfs.reindex(variable_names, level="Variable", axis=1)
            all_dfs = all_dfs.rename(
                columns={
                    variable_names[0]: "x",
                    variable_names[1]: "y",
                    variable_names[2]: "c",
                },
                level="Variable",
            )

            # Output to CSV
            all_dfs.to_csv(os.path.join(output_dir, f"person{i}.csv"))
