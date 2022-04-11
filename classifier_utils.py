import pandas as pd 
import csv

def expand_acronyms(df, col_name, acronym_csv_name):
    """Returns a dataframe with a specific column's acronyms expanded
        
        Parameters:
            df (DataFrame): A dataframe with a column of text data
            col_name (string): The name of the column with text
            acronym_csv_name (string): The name of the csv file containing an 'Acronyms'
                column with acronyms and a 'Definition' column containing the acronym expansions.
            
        Returns:
            df (DataFrame): A dataframe with the expanded acronyms
    """
    
    df = df.copy()
    with open(acronym_csv_name, newline='') as acronym_csv_file:
        csvreader_server = csv.DictReader(acronym_csv_file)
        for row in csvreader_server:
            acronym = row["Acronyms"]
            definition = row["Definition"]
            if definition == '' or definition == '?':
                continue
            df[col_name] = df[col_name].str.replace(acronym, definition)
    return df