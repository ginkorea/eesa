import pandas as pd
import os
from llm.util import *


def combine_csv_files(directory_path):
    # Step 1: Get a list of all CSV file paths in the directory
    csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]
    cyan(csv_files)

    # Step 2: Load each CSV file into a separate DataFrame
    dataframes = []
    for csv_file in csv_files:
        csv_path = os.path.join(directory_path, csv_file)
        df = pd.read_csv(csv_path, sep='|')  # Adjust sep if needed based on your file format
        dataframes.append(df)

    # Step 3: Concatenate all the DataFrames into a single combined DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Step 4: Save the combined DataFrame to a new CSV file
    output_file = f"{directory_path}_labeled.csv"
    combined_df.to_csv(output_file, index=False, sep='|')  # Adjust sep if needed based on your preference



# Example usage:
combine_csv_files('gold')
