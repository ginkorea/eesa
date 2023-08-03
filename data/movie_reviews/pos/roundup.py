import os
import pandas as pd

# Get the current directory path
current_directory = os.getcwd()

# Initialize an empty list to store the data
data = []

# Function to read the text files and extract sentences
def read_files_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            with open(os.path.join(directory_path, filename), 'r') as file:
                content = file.read()
                data.append({'sentiment': '1', 'sentence': content})  # Append sentiment and content of the file as a dictionary

# Call the function to read the files and extract sentences
read_files_in_directory(current_directory)

# Create a DataFrame from the data list
df = pd.DataFrame(data)

# Write the DataFrame to a CSV file
csv_file_path = os.path.join(current_directory, 'pos_movies.csv')
df.to_csv(csv_file_path, index=False, sep='|')

print('CSV file created successfully!')
