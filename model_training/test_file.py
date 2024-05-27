import pandas as pd

def load_and_save_csv(input_file, output_file, rows=5000):
    # Load the CSV file
    df = pd.read_csv(input_file)
    
    # Keep only the first 5000 rows
    df = df.head(rows)
    
    # Save the new CSV file
    df.to_csv(output_file, index=False)
    
    print(f"File saved as {output_file}")

# Usage
input_file = '/Users/maxmeiners/Documents/GitHub/2023-24c-fai2-adsai-MaxMeiners/Datasets_new/emotions_all.csv'
output_file = '/Users/maxmeiners/Downloads/model/test_emotions_eval'  # Replace with your output file name
load_and_save_csv(input_file, output_file)
