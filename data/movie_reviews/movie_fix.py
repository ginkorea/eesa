import os
import pandas as pd


# Function to read the contents of all .txt files in a directory and concatenate them into a single string
def read_files_in_directory(directory):
    file_contents = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                file_content = file.read().replace('\n', ' ')
                file_content = file_content.replace('|', '-')
                file_contents.append(file_content)
    return file_contents


# Function to create a DataFrame with 'sentiment' and 'sentence' columns
def create_dataframe(text, sentiment):
    data = {'sentiment': sentiment, 'sentence': text}
    df = pd.DataFrame(data)
    return df


# Main function to process all .txt files in the directory
def process_directory(directory, sentiment):
    sentences = read_files_in_directory(directory)
    numb_records = len(sentences)
    sentiment = [sentiment] * numb_records
    df = create_dataframe(sentences, sentiment)
    return df


def do_work():
    # Replace 'your_directory_path' with the actual path to the directory containing the .txt files
    pos_directory = 'pos'
    neg_directory = 'neg'

    # Process positive and negative directories separately
    positive_df = process_directory(pos_directory, 1)
    negative_df = process_directory(neg_directory, 0)

    # Concatenate the DataFrames for positive and negative directories
    result_df = pd.concat([positive_df, negative_df], ignore_index=True)

    # Save the final DataFrame to a CSV file
    output_csv = 'new_movies.csv'
    result_df.to_csv(output_csv, index=False, sep='|')
    print("saving...")


do_work()
