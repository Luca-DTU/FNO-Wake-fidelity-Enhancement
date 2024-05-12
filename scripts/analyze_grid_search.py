""" This script is used to analyze the results of a grid search experiment. 
It reads the main.log file in each subfolder of a given directory and extracts the test loss value. 
It then reads the overrides.yaml file in each subfolder to extract the hyperparameters used in the experiment. 
The results are stored in a pandas DataFrame and printed to the console. """
import os
import pandas as pd
import re
# print all df without truncation
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def analyse_multirun(path, log_file_name, target_measure, column_names=None):
    # Create an empty list to store the dictionaries
    data_list = []

    # Loop through each subfolder in the directory
    for folder in os.listdir(path):
        # Check if the current item is a directory
        if os.path.isdir(os.path.join(path, folder)):
            # Define the path to the main.log file in the current subfolder
            log_path = os.path.join(path, folder, log_file_name)
            # Open the log file and read its contents
            with open(log_path, "r") as f:
                log_contents = f.read()
            # Find the line containing the target measure value
            try:
                measure_line = [line for line in log_contents.split("\n") if target_measure in line][0]
            except IndexError as e:
                print(f"Error in {folder}")
                print(e)
                continue
            # Extract the target measure value from the line
            measure_value = float(re.search(r"'l2': (\d+\.\d+)", measure_line).group(1))
            # measure_value = float(re.search(r"Test loss l2: (\d+\.\d+)", measure_line).group(1))
            # Define the path to the overrides.yaml file in the current subfolder
            overrides_path = os.path.join(path, folder, ".hydra", "overrides.yaml")
            # Open the overrides file and read its contents
            with open(overrides_path, "r") as f:
                overrides_contents = f.read()
            # Extract the relevant parameters from the overrides file
            params = {}
            for line in overrides_contents.split("\n"):
                if "=" in line:
                    key, value = line.split("=")
                    params[key[2:]] = value.strip()
            params[target_measure] = measure_value
            params["Folder"] = folder
            # Append the dictionary to the list
            data_list.append(params)

    # Create a dataframe from the list of dictionaries
    df = pd.DataFrame(data_list)

    # Update the column names if necessary
    if column_names is not None:
        df = df.rename(columns=dict(zip(df.columns[:-2], column_names)))

    # Sort the dataframe by folder name and test loss
    df = df.sort_values(by=target_measure)
    df = df.set_index("Folder")
    return df
if __name__ == "__main__":
    # Define the path to the directory containing the subfolders
    # path = "multirun/2024-04-26/17-08-59"
    # path = "multirun/2024-05-01/18-21-02"
    # path = "multirun/2024-05-03/16-36-21"
    # path = "multirun/2024-05-04/14-23-56"
    # path = "multirun/2024-05-10/13-50-57"
    # path = "multirun/2024-04-27/15-16-35"
    path = "multirun/2024-05-11/12-37-11"
    df = analyse_multirun(path, "main.log", "Test loss: {'l2'"
                          )
    print(df)
    print(df.to_latex(index = False,float_format="%.4f"))
