import pandas as pd

def read_and_clean(filepath):
    
    try:
        df = pd.read_csv(filepath)

    except FileNotFoundError:
        print("Path to the file is wrong")
        return None
    
    return df

if __name__ == "__main__":
    print(f'Starting preprocessing data')

    filepath = '../data/heart.csv'

    funcoutput = read_and_clean(filepath)

    print(funcoutput)