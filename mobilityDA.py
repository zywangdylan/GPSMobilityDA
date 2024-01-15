import pandas as pd
import numpy as np
import glob
import os
import osmnx as ox
from shapely.geometry import Point
from tqdm import tqdm
from datetime import datetime
from geopy.geocoders import Nominatim
import geopandas as gpd
import contextily as ctx
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN


# Global Parameters
# Length of the progress bar in the console
bar_length = 120

# Base directory path of data, with all .plt files
base_dir = "./Data"

# Variables for data transforming
place_name = "Beijing, China"
place_boundary = ox.geocode_to_gdf(place_name)

start_date = "2010-01-01"
end_date = "2010-06-30"

night_start_hour = 20
night_end_hour = 4

# Path for csv storage
raw_csv_path = "./raw_trajectory_df.csv"
filtered_csv_path = "./filter_trajectory_df.csv"
night_csv_path = "./night_trajectory_df.csv"
day_csv_path = "./day_trajectory_df.csv"

matplotlib.rcParams['font.family'] = 'Songti SC'


def save_csv(df, save_path, chunk_size=1000):
    """
    Save a pandas DataFrame to a CSV file with a progress bar.

    Parameters:
        df (pandas.DataFrame): The DataFrame to be saved.
        save_path (str): The path, including the file name, where the CSV
        file will be saved.
        chunk_size (int): The number of rows per chunk to write at a time.

    Returns:
        None
    """
    try:
        # Determine the total number of chunks
        num_chunks = len(df) // chunk_size + bool(len(df) % chunk_size)

        with open(save_path, 'w', newline='', encoding='utf-8') as file:
            for i in tqdm(range(num_chunks), desc=f'Saving to {save_path}',
                          ncols=bar_length):
                start = i * chunk_size
                end = start + chunk_size

                # Write header only for the first chunk
                header = i == 0
                df.iloc[start:end].to_csv(file, mode='a', header=header,
                                          index=False)

    except Exception as e:
        print(f"Error saving DataFrame to CSV: {e}")


def load_csv(load_path):
    """
    Read a CSV file into a pandas DataFrame.

    Parameters:
        load_path (str): The path to the CSV file to be read.

    Returns:
        pandas.DataFrame: DataFrame containing the data from the CSV file.
    """
    try:
        df = pd.read_csv(load_path)
        print(f"CSV file successfully read from {load_path}")
        return df
    except FileNotFoundError:
        print(f"File not found: {load_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None


def read_plt(file_path):
    """
    Read a .plt file and return its contents as a pandas DataFrame.

    Parameters:
        file_path (str): The file path to the .plt file that needs to be read.

    Returns:
        pandas.DataFrame: A DataFrame with the columns: 'Latitude',
        'Longitude', 'Altitude', 'Zero', 'Days', 'Date', and 'Time'. The
        'Date' and 'Time' columns are combined into a single 'Datetime' column.
    """

    # Define column names based on the data structure
    # Field 1: Latitude in decimal degrees.
    # Field 2: Longitude in decimal degrees.
    # Field 3: All set to 0 for this dataset.
    # Field 4: Altitude in feet(-777 if not valid).
    # Field 5: Date - number of days(with fractional part) that have
    #   passed since 12 / 30 / 1899.
    # Field 6: Date as string.s
    # Field 7: Time as a string.
    column_names = ['latitude', 'longitude', 'zero', 'altitude', 'days',
                    'date', 'time']

    # Skip the first 6 header lines and read the data
    df = pd.read_csv(file_path, skiprows=6, header=None, names=column_names)

    return df


def read_all_plt():
    """
    Read all .plt files from a specified base directory and its subdirectories,
    combining them into a single pandas DataFrame.

    The base directory must be defined globally or within the scope accessible
    to this function as 'base_dir'.

    Returns:
        pandas.DataFrame: A concatenated DataFrame containing the data from
        all .plt files found in the directory structure. Each row in the
        DataFrame corresponds to a row from the .plt files.
    """

    if os.path.exists(raw_csv_path):
        print(f"CSV file found at {raw_csv_path}. Reading the DataFrame "
              f"from CSV.")
        return load_csv(raw_csv_path)

    all_files = glob.glob(os.path.join(base_dir, '**/Trajectory/*.plt'),
                          recursive=True)
    df_list = []

    for file in tqdm(all_files, desc="Reading all .plt files",
                     ncols=bar_length):
        # The user_id is the name of the dict
        user_id = os.path.basename(os.path.dirname(os.path.dirname(file)))
        df = read_plt(file)
        df['user_id'] = user_id
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    save_csv(combined_df, raw_csv_path)

    return combined_df


def filter_by_date(df, after_date, before_date):
    """
    Filter a DataFrame to include rows where the date is before a specified
    date.

    Parameters:
        df (pandas.DataFrame): The DataFrame to filter, which must contain
        a date column.
        after_date (str or datetime-like): The date after which rows should be
        included. This can be a string or any datetime-like object that
        pandas can interpret as a date.
        before_date (str or datetime-like): The date before which rows
        should be included. This can be a string or any datetime-like
        object that pandas can interpret as a date.

    Returns:
        pandas.DataFrame: A DataFrame containing only the rows from
        the original DataFrame where the date is before the specified
        before_date.
    """
    # Ensure the 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'])

    after_date = pd.to_datetime(after_date)
    before_date = pd.to_datetime(before_date)

    tqdm.pandas(desc="Filtering by date", ncols=bar_length)
    mask = df['date'].progress_apply(lambda x: after_date <= x <= before_date)

    return df[mask]


def check_if_in_target_region(latitude, longitude):
    """
    Check if a given geographical point (latitude and longitude) is within
    the boundary of place_name.

    Parameters:
        latitude (float): Latitude of the point to check.
        longitude (float): Longitude of the point to check.

    Returns:
        bool: True if the point is within the boundary of Beijing, False
        otherwise.
    """
    point = Point(longitude, latitude)
    is_in = place_boundary.contains(point).any()

    return is_in


def filter_by_region(df):
    """
    Filter a DataFrame based on whether each row's geographic location
    falls within a target region.

    Parameters:
       df (pandas.DataFrame): DataFrame with 'Latitude' and 'Longitude'
       columns.

    Returns:
       pandas.DataFrame: Filtered DataFrame containing only rows within the
       target region.
    """
    df_filtered = df.copy()

    # Apply the check_if_in_target_region function to each row
    tqdm.pandas(desc="Filtering by region", ncols=bar_length)
    df_filtered['is_in_target_region'] = df_filtered.progress_apply(
        lambda row: check_if_in_target_region(row['latitude'],
                                              row['longitude']), axis=1
    )

    # Filter the DataFrame based on the 'is_in_target_region' column
    return df_filtered[df_filtered['is_in_target_region']]


def filter_users_by_min_pings(df, min_pings):
    """
    Filter out users from the DataFrame who have fewer than a specified
    number of pings.

    Parameters:
        df (pandas.DataFrame): DataFrame containing the user trajectory data.
                           It must have a 'user_id' column.
        min_pings (int): Minimum number of pings required for a user to be
        included in the returned DataFrame.

    Returns:
        pandas.DataFrame: A DataFrame containing only the users who have at
        least 'min_pings' number of pings.
    """
    # Group by 'user_id' and count the number of pings for each user
    ping_counts = df.groupby('user_id').size()

    # Identify users who meet the minimum ping count
    users_to_keep = ping_counts[ping_counts >= min_pings].index

    # Filter and return the DataFrame
    return df[df['user_id'].isin(users_to_keep)]


# user_id ping_counts
# 45       6358
# 46      16451
# 65       2619
# 79      11243
# 113     24531
# 114     12275
# 128    176240
# 142     55839
# 153    203334
# 163     57146
# 169     48321
# Data Pre-processing
if os.path.exists(filtered_csv_path):
    print(f"CSV file found at {filtered_csv_path}. Reading the DataFrame "
          f"from CSV.")
    filter_df = load_csv(filtered_csv_path)
else:
    print("Start filtering dataframe by region and date")
    raw_df = read_all_plt()
    filter_df = filter_by_region(filter_by_date(raw_df, start_date, end_date))

    # Add a datetime timestamp in the df
    filter_df['datetime'] = pd.to_datetime(filter_df['date'] + ' ' + filter_df[
        'time'])

    # Drop the boolean column after filtering
    filter_df = filter_df.drop('is_in_target_region', axis=1)
    save_csv(filter_df, filtered_csv_path)


def is_night_time(timestamp):
    """
    Check if a given timestamp is during the night.

    Parameters:
        timestamp (datetime): The timestamp to be checked.

    Returns:
        bool: True if the timestamp is during the night, False otherwise.
    """
    if pd.isnull(timestamp):
        return False

    timestamp = pd.to_datetime(timestamp, errors='coerce')

    # Define night time range
    night_start = timestamp.replace(hour=night_start_hour, minute=0,
                                    second=0, microsecond=0)
    night_end = timestamp.replace(hour=night_end_hour, minute=0, second=0,
                                  microsecond=0)

    # Adjust for timestamps after midnight
    if timestamp.hour < 6:
        night_start = night_start - pd.Timedelta(days=1)

    return night_start <= timestamp <= night_end or timestamp <= night_end


# Parse and transform the data to be able to identify pings that occurred
# at night
if os.path.exists(night_csv_path) and os.path.exists(day_csv_path):
    print(f"CSV file found at {night_csv_path} and {day_csv_path}. Reading "
          f"the DataFrame from CSV.")
    night_pings = load_csv(night_csv_path)
    day_pings = load_csv(day_csv_path)
else:
    tqdm.pandas(desc="Splitting pings during night/day", ncols=bar_length)
    filter_df['is_night'] = filter_df['datetime'].progress_apply(is_night_time)
    night_pings = filter_df[filter_df['is_night']]
    day_pings = filter_df[~filter_df['is_night']]

    # Drop the temporary is_night column
    night_pings = night_pings.drop('is_night', axis=1)
    day_pings = day_pings.drop('is_night', axis=1)

    # Save to csv file
    save_csv(night_pings, night_csv_path)
    save_csv(day_pings, day_csv_path)


def convert_to_geodf(df):
    """
    Convert df to geoDf

    Parameters:
       df (pandas.DataFrame): DataFrame with 'Latitude' and 'Longitude'
       columns.

    Returns:
       GeoDataFrame
    """
    return gpd.GeoDataFrame(
        df, geometry=[Point(xy) for xy in zip(df.longitude, df.latitude)],
        crs="EPSG:4326")


def plot_pings(gdf_day, gdf_night):
    """
    Using the geoDataframe to plot the tracks on map

    Parameters:
       gdf_day(gdf): gdf pings at day
       gdf_night(gdf): gdf pings at night

    Returns:
       None
    """
    # Get a list of unique users from both day and night GeoDataFrames
    unique_users = pd.concat(
        [gdf_day['user_id'], gdf_night['user_id']]).unique()

    # Create a dictionary to map each user_id to a color
    colormap = plt.cm.Dark2
    color_map = {user_id: colormap(i % colormap.N) for i, user_id in
                 enumerate(unique_users)}

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    # Plot for Day Pings
    axes[0, 0].set_title('Day Pings')
    # Plot each user's pings in different colors
    for user_id, user_group in gdf_day.groupby('user_id'):
        gpd.GeoDataFrame(user_group).plot(ax=axes[0, 0], color=color_map[
            user_id], label=str(user_id), markersize=5, alpha=0.2)
    ctx.add_basemap(axes[0, 0], crs=gdf_day.crs.to_string(),
                    source=ctx.providers.OpenStreetMap.Mapnik)
    axes[0, 0].legend(title='User ID')

    # Plot for Night Pings
    axes[0, 1].set_title('Night Pings')
    # Plot each user's pings in different colors
    for user_id, user_group in gdf_night.groupby('user_id'):
        gpd.GeoDataFrame(user_group).plot(ax=axes[0, 1], label=str(user_id),
                                          color=color_map[user_id],
                                          markersize=5, alpha=0.2)
    ctx.add_basemap(axes[0, 1], crs=gdf_night.crs.to_string(),
                    source=ctx.providers.OpenStreetMap.Mapnik)
    axes[0, 1].legend(title='User ID')

    # Plot Day Pings Density Heatmap
    axes[1, 0].set_title('Day Pings Density')
    sns.kdeplot(x=gdf_day.geometry.x, y=gdf_day.geometry.y,
                ax=axes[1, 0], cmap="Greens", fill=True, alpha=0.7)
    ctx.add_basemap(ax=axes[1, 0], crs=gdf_day.crs.to_string(),
                    source=ctx.providers.OpenStreetMap.Mapnik)

    # Plot Night Pings Density Heatmap
    axes[1, 1].set_title('Night Pings Density')
    sns.kdeplot(x=gdf_night_pings.geometry.x, y=gdf_night_pings.geometry.y,
                ax=axes[1, 1], cmap="Blues", fill=True, alpha=0.7)
    ctx.add_basemap(ax=axes[1, 1], crs=gdf_night_pings.crs.to_string(),
                    source=ctx.providers.OpenStreetMap.Mapnik)

    for ax in axes.flatten():
        ax.axis('off')

    # Display the plot
    plt.tight_layout()
    plt.show()


selected_users = [45, 79, 113, 114, 169]
gdf_day_pings = convert_to_geodf(day_pings)
gdf_night_pings = convert_to_geodf(night_pings)

# Filter for selected users
gdf_day_pings_selected = gdf_day_pings[gdf_day_pings['user_id'].isin(
    selected_users)]
gdf_night_pings_selected = gdf_night_pings[gdf_night_pings['user_id'].isin(
    selected_users)]
# plot_pings(gdf_day_pings_selected, gdf_night_pings_selected)


def get_address_from_coord(lat, lng):
    geolocator = Nominatim(user_agent="myGeocodeApp")
    location = geolocator.reverse((lat, lng), exactly_one=True)
    address = location.address if location else "Address not found"
    return address


def plot_user_home_pings(df, home_location, home_loc_address):
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude)
    )
    gdf.set_crs(epsg=4326, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot all pings
    gdf.plot(ax=ax, color='blue', markersize=5, alpha=0.5, label='Pings')

    # Highlight the home location
    home_point = gpd.GeoDataFrame(
        [{'geometry': gpd.points_from_xy([home_location.longitude],
                                         [home_location.latitude])[0]}],
        crs='EPSG:4326'
    )

    sns.kdeplot(x=gdf.geometry.x, y=gdf.geometry.y,
                ax=ax, cmap="Blues", fill=True, alpha=0.7)

    home_point.plot(ax=ax, color='red', markersize=50, label='Home Location')

    # Add basemap
    ctx.add_basemap(ax, crs=gdf.crs.to_string(),
                    source=ctx.providers.OpenStreetMap.Mapnik)

    # Set axis
    ax.set_axis_off()

    # Add the address below the plot
    plt.gcf().text(0.5, 0.1, home_loc_address, ha='center', va='bottom',
                   fontsize=12)

    # Add legend
    ax.legend()

    plt.show()


# DBSCAN to find the home location for a user
user_id_of_interest = 114
night_ping_for_user = night_pings[night_pings['user_id'] ==
                                  user_id_of_interest].copy()

# Convert latitude and longitude to radians
coords = night_ping_for_user[['latitude', 'longitude']].to_numpy()
coords_in_radians = np.radians(coords)

kms_per_radian = 6371.0088  # constant to convert radians to kilometers
epsilon = 100 / 1000 / kms_per_radian  # 100 meters in radians

db = DBSCAN(eps=epsilon, min_samples=5, algorithm='ball_tree',
            metric='haversine').fit(coords_in_radians)

night_ping_for_user.loc[:, 'cluster'] = db.labels_

# Find the cluster with the most pings
home_cluster = night_ping_for_user['cluster'].value_counts().idxmax()
# Get the average (mean) coordinates of the home cluster
home_location = (night_ping_for_user[night_ping_for_user['cluster'] ==
                                     home_cluster][['latitude',
                                                   'longitude']].mean())
home_loc_address = get_address_from_coord(home_location['latitude'],
                                          home_location['longitude'])
plot_user_home_pings(night_ping_for_user, home_location, home_loc_address)

# Next Step:

# 8. Describe this algorithm in detail and how it will tackle problems like
# data sparsity, people staying in hotels or other residences many nights,
# and people moving to a different location permanently. What is the
# complexity of this algorithm?
# 9. Implement this algorithm for a few users and provide the necessary
# code. Would this algorithm work the same if the users had 10% of the data
# they have? Compare the results.
# 10. Would it be challenging to scale this to multiple users? Provide a few
#  plots to support your reasoning about your findings.

