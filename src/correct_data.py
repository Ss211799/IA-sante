from pathlib import Path
print("RUNNING:", Path(__file__).resolve())

import pandas as pd
from .config import DATA_DIR

def remove_protein(df):
    return df.drop(columns=["total_protein"], errors="ignore")

def correct_ser_bilir_unit_104(df):
    if "id" in df.columns and "ser_bilir" in df.columns:
        df.loc[df["id"] == 104, "ser_bilir"] = df.loc[df["id"] == 104, "ser_bilir"] / 100
    return df

def remove_date_diag(df):
    return df.drop(columns=["date_diag"], errors="ignore")

def impute_missing_with_median(df):
    continuous_vars = ["ser_bilir", "serChol", "albumin", "alkaline", "SGOT", "platelets", "prothrombin"]
    for col in continuous_vars:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df

def impute_missing_with_mode(df):
    categorical_vars = ["ascites", "hepatomegaly", "spiders", "edema", "histologic"]
    for col in categorical_vars:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode(dropna=True)[0])
    return df

def replace_label_2_by_0(df):
    if "label" in df.columns:
        df.loc[df["label"] == 2, "label"] = 0
    return df

def main():
    # Load raw data
    df = pd.read_csv(DATA_DIR / "clinical_data_pbc.csv")

    # Clean
    df = remove_protein(df)
    df = correct_ser_bilir_unit_104(df)
    df = remove_date_diag(df)
    df = impute_missing_with_median(df)
    df = impute_missing_with_mode(df)
    df = replace_label_2_by_0(df)

    # Save cleaned data
    output_path = DATA_DIR / "clinical_data_pbc_cleaned.csv"
    df.to_csv(output_path, index=False)

    print(df.head())
    print(df.info())

if __name__ == "__main__":
    main()
