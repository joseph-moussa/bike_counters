import streamlit as st
import pandas as pd
import numpy as np
import folium
import seaborn as sns
import matplotlib.pyplot as plt

# Data can be downloaded from the links in the readme file
data = pd.read_parquet("C:\Users\Joseph Moussa\Desktop\X\4A\Tooling for data scientist\streamlit\bike-counters\train.parquet")
'''
data_url = "https://github.com/ramp-kits/bike_counters/releases/download/v0.1.0/train.parquet"
response = requests.get(data_url)
parquet_content = response.content
parquet_io = BytesIO(parquet_content)
data = pd.read_parquet(parquet_io)
'''

# Function to visualize data on a map
def visualize_data_on_map(data):
    m = folium.Map(location=data[["latitude", "longitude"]].mean(axis=0), zoom_start=13)

    for _, row in (
        data[["counter_name", "latitude", "longitude"]]
        .drop_duplicates("counter_name")
        .iterrows()
    ):
        folium.Marker(
            row[["latitude", "longitude"]].values.tolist(), popup=row["counter_name"]
        ).add_to(m)
    
    return m

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

# Streamlit app
def main():
    st.title("Data Visualization with Streamlit")

    st.header("Visualize Data on a Map")
    map_data = visualize_data_on_map(data)
    st.write("Map of Counter Locations")
    st.write(map_data)

    st.header("Aggregate Data and Plot")
    aggregate_and_plot_data(data)

    st.header("Distribution of the Target Variable")
    display_target_variable_distribution(data)

    st.header("Logarithmic Transformation")
    display_logarithmic_transformation(data)

if __name__ == '__main__':
    main()
