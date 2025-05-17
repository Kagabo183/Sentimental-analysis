import pandas as pd

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath, encoding='latin1')
    df = df[df["Translated Comments (English)"].notnull()].copy()
    df["Translated Comments (English)"] = df["Translated Comments (English)"].str.strip()
    df = df[df["Translated Comments (English)"] != ""]
    df.reset_index(drop=True, inplace=True)
    return df
