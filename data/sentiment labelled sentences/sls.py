import pandas as pd
import re
from util import *


def txt2csv(input_file):
    # Read the text file line by line and extract text and label using regular expressions
    data = []
    file_name = input_file.replace('.txt', '.csv')
    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespaces
            match = re.match(r'^(.*?)(\d)$', line)  # Regular expression to extract text and label
            if match:
                text, label = match.groups()
                data.append([int(label), text.strip()])  # Swap the order of columns here
    # Convert the list of data into a DataFrame
    df = pd.DataFrame(data, columns=['sentiment', 'sentence'])  # Rename the columns here
    # Save the DataFrame to a CSV file
    df.to_csv(file_name, index=False, sep='|')
    green("saving %s" % file_name)


txt2csv('amazon_cells_labelled.txt')

