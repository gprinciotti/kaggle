"""
Módulo com funções utilitárias para o projeto.
"""

# -----------------------------------------------------------------------------
# Importações básicas
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
from IPython.display import display

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

    # summarise the dataset
    print('Summary:')
    display(df.describe().round(2))

def rolling_average(df, time_var, outcomes, window = 7):
    """
    Calculate the rolling average of specified outcome variables over a given time window.
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    time_var : str
        The name of the column representing the time variable to group by.
    outcomes : str or list of str
        The name(s) of the outcome variable(s) for which the rolling average is calculated.
        If a single string is provided, it will be converted to a list.
    window : int, optional, default=7
        The size of the rolling window to compute the average.
    Returns:
    --------
    pandas.DataFrame
        If a single outcome variable is provided, returns a DataFrame with the rolling average
        for that variable. If multiple outcome variables are provided, returns a melted DataFrame
        with columns for the time variable, variable names, and their corresponding rolling averages.
    Notes:
    ------
    - The rolling average is computed using a minimum of 1 period, so the first few rows
        will not have NaN values even if the window size is larger than the available data.
    - When multiple outcome variables are provided, the result is reshaped into a long format
        using `pandas.melt`.
    Example:
    --------
    >>> import pandas as pd
    >>> data = {'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
    ...         'sales': [100, 200, 300],
    ...         'profit': [50, 80, 120]}
    >>> df = pd.DataFrame(data)
    >>> rolling_average(df, time_var='date', outcomes=['sales', 'profit'], window=2)
    """

    if isinstance(outcomes, str):
        outcomes = [outcomes]
    
    grouped_df = (
        df
        .groupby(time_var, as_index=False)
        .agg({var: 'mean' for var in outcomes})
    )

    rolling_avg = grouped_df.copy()

    for var in outcomes:
        rolling_avg[var] = (
            grouped_df[var]
            .rolling(window=window, min_periods=1)
            .mean()
        )

    if len(outcomes) > 1:
        return rolling_avg.melt(id_vars=time_var, var_name='variables', value_name='values')
    else:
        return rolling_avg
