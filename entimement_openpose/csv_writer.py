import os
import pandas as pd


class CSVWriter:

    def writeCSV(dataframes, output_dir):
        n_people = n_people = len(dataframes[0].columns) // 3

        for i in range(n_people):
            person_dfs = [df.iloc[:, i*3:i*3+3] for df in dataframes]
            all_dfs = pd.concat(person_dfs, keys=range(len(person_dfs)))

            all_dfs.to_csv(os.path.join(output_dir, f'person{i}.csv'))
