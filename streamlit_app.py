import streamlit as st
import pandas as pd
import numpy as np
import folium
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# Data can be downloaded from the links in the readme file
train_data = pd.read_parquet("data/train.parquet")
train_data.drop(columns=["SC'], inplace=True)
test_data = pd.read_parquet("data/test.parquet")
test_data.drop(columns=["SC'], inplace=True)

# Function to visualize data on a map
def visualize_data_on_map(data):
    counter_locations = data.groupby("counter_name")[["latitude", "longitude"]].mean().reset_index()
    st.map(counter_locations)

# Function to aggregate data and plot
def aggregate_and_plot_data(data):
    mask = data["counter_name"] == "Totem 73 boulevard de Sébastopol S-N"
    aggregated_data = data[mask].groupby(pd.Grouper(freq="1w", key="date"))[["bike_count"]].sum()
    
    # Plot the aggregated data
    plt.figure(figsize=(10, 6))
    plt.plot(aggregated_data.index, aggregated_data["bike_count"])
    plt.xlabel("Date")
    plt.ylabel("Total Bike Count")
    plt.title("Bike Count Over Time")
    st.pyplot()

# Function to display the distribution of the target variable
def display_target_variable_distribution(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data, x="bike_count", kde=True, bins=50, ax=ax)
    plt.xlabel("Bike Count")
    plt.ylabel("Frequency")
    plt.title("Distribution of Bike Count")
    st.pyplot()

# Function to display the distribution after logarithmic transformation
def display_logarithmic_transformation(data):
    data["log_bike_count"] = data["bike_count"].apply(lambda x: max(1, x)).apply(lambda x: np.log(1 + x))
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data, x="log_bike_count", kde=True, bins=50, ax=ax)
    plt.xlabel("Log(Bike Count)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Logarithm of Bike Count")
    st.pyplot()

# Date encoding function:
def encode_dates(X):
    '''
    Encode date-related features from a DataFrame into separate columns.

    Parameters:
        X (DataFrame): The input DataFrame containing a 'date' column to be encoded.

    Returns:
        DataFrame: A modified DataFrame with additional columns for year, month, day, weekday, and hour.
    '''
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])

# Function to visualize predictions
def visualize_predictions(regressor, X_test, y_test, counter_name = "Totem 73 boulevard de Sébastopol S-N", start_date = "2021/09/01", end_date = "2021/09/08"):
    mask = (
    (X_test["counter_name"] == counter_name)
    & (X_test["date"] > pd.to_datetime(start_date))
    & (X_test["date"] < pd.to_datetime(end_date))
    )
    df_viz = X_test.loc[mask].copy()
    df_viz["bike_count"] = np.exp(y_test[mask.values]) - 1
    df_viz["bike_count (predicted)"] = np.exp(regressor.predict(X_test[mask])) - 1
    fig, ax = plt.subplots(figsize=(12, 4))
    df_viz.plot(x="date", y="bike_count", ax=ax)
    df_viz.plot(x="date", y="bike_count (predicted)", ax=ax, ls="--")
    ax.set_title("Predictions with Ridge")
    ax.set_ylabel("bike_count")
    plt.show()

# Streamlit app
def main():
    # Add title
    st.title('Analysis of the Bike traffic in Paris')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.header("Map Visualization of the data")
    st.write("#### Map of Counter Locations in Paris")
    visualize_data_on_map(train_data)

    st.header("Smoothed Temporal Distribution of the most frequented bike counter")
    st.write(" #### Data Aggregated by Week")
    aggregate_and_plot_data(train_data)

    st.header("Distribution of the Target Variable - Bike Count")
    display_target_variable_distribution(train_data)
    st.write("If we look at the distribution of the target variable, we can see that it is skewed and non normal. \
             A loss such as the MSE would not be appropriate since it is desined for normal error distributions. \
             One way to procede would be to transform the variable with a logarithmic transformation.")

    st.header("Logarithmic Transformation")
    display_logarithmic_transformation(train_data)

    st.header("Applying Simple Linear Regression")
    X_train, y_train = encode_dates(train_data), train_data["log_bike_count"]
    X_test, y_test = encode_dates(test_data), test_data["log_bike_count"] 
    X_train[["site_id", "counter_id"]] = X_train["counter_id"].str.split("-", expand=True)
    X_test[["site_id", "counter_id"]] = X_test["counter_id"].str.split("-", expand=True)
    regressor = Ridge()
    regressor.fit(X_train, y_train)

    st.write('Bike count predictions for one specific counter: Totem 73 boulevard de Sébastopol S-N')
    visualize_predictions(regressor, X_test, y_test)

    st.write('#### Interpretation:)

if __name__ == '__main__':
    main()
