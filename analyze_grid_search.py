
import os
import pandas as pd
import re
# print all df without truncation
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def analyse_multirun(path):
    # Create an empty dataframe to store the test loss values and other parameters
    df = pd.DataFrame(columns=["L2 Loss", "Batch Size", "Number of Layers", "Non Linearity", "Data Source", "Loss type"])

    # Loop through each subfolder in the directory
    for folder in os.listdir(path):
        # Check if the current item is a directory
        if os.path.isdir(os.path.join(path, folder)):
            # Define the path to the main.log file in the current subfolder
            log_path = os.path.join(path, folder, "main.log")
            # Open the log file and read its contents
            with open(log_path, "r") as f:
                log_contents = f.read()
            # Find the line containing the test loss value
            try:
                loss_line = [line for line in log_contents.split("\n") if "Test loss l2:" in line][0]
            except IndexError as e:
                print(f"Error in {folder}")
                print(e)
                continue
            # Extract the test loss value from the line
            test_loss = float(loss_line.split(":")[-1])
            
            # Define the path to the overrides.yaml file in the current subfolder
            overrides_path = os.path.join(path, folder, ".hydra", "overrides.yaml")
            # Open the overrides file and read its contents
            with open(overrides_path, "r") as f:
                overrides_contents = f.read()
            # Extract the relevant parameters from the overrides file
            batch_size = int(re.findall(r"train.batch_size=(\d+)", overrides_contents)[0])
            n_layers = int(re.findall(r"TFNO.n_layers=(\d+)", overrides_contents)[0])
            non_linearity = re.findall(r"TFNO.non_linearity=(\w+)", overrides_contents)[0]
            data_source = re.findall(r"data_source.train_args.inputs=\[(.*?)\]", overrides_contents)[0]
            loss = re.findall(r"train.loss=(\w+)", overrides_contents)[0]
            # Add the test loss value and other parameters to the dataframe
            df.loc[folder] = [test_loss, batch_size, n_layers, non_linearity, data_source, loss]

    # set index as int and sort it
    df.index = df.index.astype(int)
    df = df.sort_index()
    # sort by Test Loss
    df = df.sort_values(by="L2 Loss")
    return df
if __name__ == "__main__":
    # Define the path to the directory containing the subfolders
    path = "multirun/2024-04-26/17-08-59"
    df = analyse_multirun(path)
    print(df)
    print(df.to_latex(index = False,float_format="%.4f"))
