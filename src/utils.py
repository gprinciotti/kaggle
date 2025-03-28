"""
Módulo com funções utilitárias para o projeto.
"""

# -----------------------------------------------------------------------------
# Importações básicas
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Análise exploratória de dados
# -----------------------------------------------------------------------------
def quick_eda(df):
    """
    Perform a quick exploratory data analysis (EDA) on a pandas DataFrame.
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to analyze.
    Outputs:
    --------
    - Prints the shape of the DataFrame (number of rows and columns).
    - Prints the number of duplicate rows in the DataFrame.
    - Displays the first few rows of the DataFrame as a sample.
    - Prints a summary of data types, non-missing counts, missing counts, 
      and the percentage of missing values for each column.
    Notes:
    ------
    This function uses `display()` to show the DataFrame and summary table, 
    which is particularly useful in Jupyter Notebook environments.
    """
    
    # dataframe shape
    print(f'Shape: {df.shape[0]} rows and {df.shape[1]} columns')

    # duplicates check
    dupes = df.duplicated().sum()
    print(f'Duplicates check: {dupes} duplicate rows found\n')

    # sample data
    print('Sample data:')
    display(df.head())

    # data types and (non-)missing count
    print('Data types and missing count:')    
    info_df = pd.DataFrame({
        'dtype': df.dtypes,
        'non_missing': df.count(),
        'missing': df.isnull().sum(),
        'missing_pct': round(df.isnull().mean() * 100, 2)
    })
    display(info_df)
