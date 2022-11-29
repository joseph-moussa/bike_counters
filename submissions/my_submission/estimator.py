from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMRegressor

def public_works(temp):
    """Returns if there are public works on a counter on a given date.

    Parameters
    ----------
    temp: (string, TimeStamp) tuple
    The temporary variable stocking both counter name and the date. See the _encode_dates function for more info.

    Returns
    -------
    result: int
        Result indicating the nature of the public works
    """
    result = 0
    # Test the value of temp[0], which will be the counter name
    if temp[0] ==  "152 boulevard du Montparnasse E-O":
        # Test the value of temp[1], which will be the date
        result = int(temp[1] >= pd.to_datetime("2021/01/25") and temp[1] <= pd.to_datetime("2021/02/25"))

    if temp[0] == "152 boulevard du Montparnasse O-E":
        result =  int(temp[1] >= pd.to_datetime("2021/01/25") and temp[1] <= pd.to_datetime("2021/02/25"))

    if temp[0] == "20 Avenue de Clichy NO-SE":
        if (temp[1] >= pd.to_datetime("2021/04/08")  and temp[1] <= pd.to_datetime("2021/07/21")):
            result = 2

    if temp[0] == "20 Avenue de Clichy SE-NO":
        if (temp[1] >= pd.to_datetime("2021/04/08")  and temp[1] <= pd.to_datetime("2021/07/21")):
            result = 2

    if temp[0] == "Voie Georges Pompidou SO-NE":
        if (temp[1] >= pd.to_datetime("2021/03/13")  and temp[1] <= pd.to_datetime("2021/04/01")):
            result = 3

    if temp[0] == "Voie Georges Pompidou NE-SO":
        if (temp[1] >= pd.to_datetime("2021/03/13")  and temp[1] <= pd.to_datetime("2021/04/01")):
            result = 3

    return result

def identify_counter(X):
    """Modifies the dataframe X to deal with holes in measurements or non available data.

    Parameters
    ----------
    X: pandas dataframe
        The features dataframe
    
    Returns
    X_transformed: pandas dataframe
        The transformed features dataframe
    """
    for i in X.index:
        if X.travaux[i] == 1:
            if X.month[i] == 1:
                X.month[i] = 12
                X.year[i] -= 1
            else:
                X.month[i] -= 1
        if X.travaux[i] == 2:
            X.month[i] = 3
        if X.travaux[i] == 3:
            X.month[i] -= 1 
    X_transformed = X 
    return X_transformed

def _encode_dates(X):
    """Adds new date-related features and returns new feature matrix X.

    Parameters
    ----------
    X: pandas dataframe
        The features dataframe.

    Returns
    -------
    X_transformed: pandas dataframe
        The dataframe with the new encoded features.
    """
    # Modify a copy of X
    X = X.copy()

    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    # Determine whethere there are public works or not
    X['temp'] = list(zip(X["counter_name"], X["date"])) # temporary column used to give indication of the presence of public works
    X["travaux"] = [public_works(temp) for temp in X["temp"]]

    # Transform X according to counters and drop the temporary column
    X = identify_counter(X)
    X.drop(columns=["temp"], inplace=True)

    # Add trigonometric features to take periodicity into account
    X.loc[:, "sin_month"] = np.sin(2 * np.pi * X["date"].dt.month / 12)
    X.loc[:, "sin_weekday"] = np.sin(2 * np.pi * X["date"].dt.weekday / 7)
    X.loc[:, "sin_day"] = np.sin(2 * np.pi * X["date"].dt.day / 30)
    X.loc[:, "sin_hour"] = np.sin(2 * np.pi * X["date"].dt.hour / 24)

    # Determine whether there is a quarantine or not
    d1, d2, d3, d4 = pd.to_datetime('2020-10-30'), pd.to_datetime('2020-12-15'), pd.to_datetime('2021-04-03'), pd.to_datetime('2021-05-03')
    quarantine = (
        ((d1 <= X['date']) & (X['date'] <= d2)) | ((d3 <= X['date']) & (X['date'] <= d4))
    )
    X["quarantine"] = np.where(quarantine, 1, 0)

    # Determine whether there is a curfew or not
    d5, d6, d7 = pd.to_datetime('2021-01-16'), pd.to_datetime('2021-03-20'), pd.to_datetime('2021-05-19')
    d8, d9 =  pd.to_datetime('2021-06-09'), pd.to_datetime('2021-06-20')
    curfew = (
        (X['date'] >= d5) & (X['date'] < d6) & ((X['hour'] >= 18) | (X['hour'] <= 6)) |
        (X['date'] >= d6) & (X['date'] < d7) & ((X['hour'] >= 19) | (X['hour'] <= 6)) |
        (X['date'] >= d7) & (X['date'] < d8) & ((X['hour'] >= 21) | (X['hour'] <= 6)) |
        (X['date'] >= d8) & (X['date'] < d9) & ((X['hour'] >= 21) | (X['hour'] <= 6))
    )
    X['curfew'] = np.where(curfew, 1, 0)

    # Drop the date column from the dataframe
    X_transformed = X.drop(columns=["date"])

    return X_transformed


def _merge_external_data(X_initial):
    """Merges existing dataframe with an external one and returns the new merged dataframe while adding new pertinent features.

    Parameters
    ----------
    X_initial: pandas dataframe
        The features dataframe to be joined.

    Returns
    -------
    X: pandas dataframe
        The merged dataframe.
    """    
    # Get the external dataframe
    file_path = Path(__file__).parent / "external_data.csv"
    df_ext = pd.read_csv(file_path, parse_dates=["date"])
    variables_to_keep = ['date', 'dd', 'tend', 'pmer', 't', 'vv', 'rr1', 'etat_sol', 'rr3', 'rr12', 'u', 'n', 'ht_neige']
    X_ext = df_ext[variables_to_keep]

    # Copy the initial dataframe
    X = X_initial.copy()

    # Merge the two dataframes
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"),  X_ext.sort_values("date").sort_values("date"), on="date"
    )

    # Calculate the distance between the counter and the Opéra
    X["distance_opera"] = np.sqrt((X["latitude"] - 48.8790) ** 2 + (X["longitude"] - 2.3378) ** 2)

    # Treat missing values in the dataframe, sort back to original order and then drop columns from merged dataframe
    for variable in variables_to_keep:
        X[variable] = X[variable].fillna(method="pad")
    X = X.sort_values("orig_index")
    X.drop(columns=["orig_index", 'counter_id', 'site_id', 'counter_installation_date', 'counter_technical_id', 'latitude', 'longitude'], inplace=True)

    return X


def get_estimator():
    """Returns the pipeline putting together the data preprocessing and the regressor.

    Parameters
    ----------
    None

    Returns
    -------
    pipe: Pipeline
        The model pipeline.
    """ 
    # Encode dates
    date_encoder = FunctionTransformer(_encode_dates)

    # Encode categorical variables
    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name"]

    # Scale numerical variables
    numeric_encoder = StandardScaler()
    numeric_cols = ['dd', 'tend', 'pmer', 't', 'vv', 'rr3', 'rr12', 'u', 'n', 'ht_neige', 'year', 'month', 'day', 'weekday', 'hour', 'sin_hour', 'sin_month', 'sin_day', 'sin_weekday', 'distance_opera', 'rr1', 'etat_sol']
    
    # Put all the preprocessing steps together in the preprocessor
    preprocessor = ColumnTransformer(
        [
            ("cat", categorical_encoder, categorical_cols),
        	("numeric", numeric_encoder, numeric_cols) 
        ]
    )
    
    # Use LGBM Regressor
    regressor = LGBMRegressor(learning_rate=0.05, boosting_type='gbdt', n_estimators=3000, max_depth=15)
    
    # Put everything together in a pipeline
    pipe = make_pipeline(
        FunctionTransformer(_merge_external_data, validate=False),
        date_encoder,
        preprocessor,
        regressor,
    )

    return pipe
