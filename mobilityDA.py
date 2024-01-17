import pandas as pd
import numpy as np
import glob
import os
import osmnx as ox
import geopandas as gpd
import contextily as ctx
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from shapely.geometry import Point
from tqdm import tqdm
from datetime import datetime
from geopy.geocoders import Nominatim
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
# Plot day & night pings for selected users
# plot_pings(gdf_day_pings_selected, gdf_night_pings_selected)


def get_address_from_coord(lat, lng):
    """
    Getting the address name for given coordinate

    Parameters:
       lat(float): latitude of the location
       lng(float): longitude of the location

    Returns:
       String: address name

    """
    geolocator = Nominatim(user_agent="myGeocodeApp")
    location = geolocator.reverse((lat, lng), exactly_one=True)
    address = location.address if location else "Address not found"
    return address


def plot_user_home_pings(df, user_loc):
    """
    Using the dataframe to plot the tracks and home for a single user

    Parameters:
        df(Pandas.dataframe): df of user's pings
        user_loc(dict): home location(lat, lng) and address name

    Returns:
        None
    """
    gdf = convert_to_geodf(df)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot all pings
    gdf.plot(ax=ax, color='blue', markersize=5, alpha=0.5, label='Pings')

    # Highlight the home location
    home_point = gpd.GeoDataFrame(
        [{'geometry': gpd.points_from_xy([user_loc['lng']],
                                         [user_loc['lat']])[0]}],
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

    fig.suptitle(f'Home location plot for user '
                 f'{df.iloc[0].user_id}', fontsize=16)

    # Add the address below the plot
    plt.gcf().text(0.5, 0.1, user_loc['address'], ha='center', va='bottom',
                   fontsize=12)

    # Add legend
    ax.legend()

    plt.show()


def plot_full_vs_reduced_data(full_df, full_loc, reduced_df, reduced_loc):
    """
    Plots two subgraphs comparing the full dataset locations with the
    reduced dataset home locations.

    This function creates a visual comparison between all user locations in
    the full dataset and the identified home locations in the reduced
    dataset. It helps in understanding the impact of data reduction on the
    accuracy of home location identification.

    Parameters:
    - full_df (DataFrame): A DataFrame containing the full dataset
    with columns ['geometry', 'user_id'], representing the complete set of
    user location data.
    - full_loc (str): A string representing the address or descriptive name
    of the area covered by the full dataset. This text will be displayed
    below the first subplot.
    - reduced_df (DataFrame): A GDataFrame similar to full_df, but containing a
    reduced set of user data. It should have the same structure as full_df.
    - reduced_loc (str): A string representing the address or descriptive
    name of the area covered by the reduced dataset. This text will be
    displayed below the second subplot.

    Returns:
    None: The function creates and displays a matplotlib figure with two
    subplots but does not return any value.
    """
    full_gdf = convert_to_geodf(full_df)
    reduced_gdf = convert_to_geodf(reduced_df)

    fig, axes = plt.subplots(1, 2, figsize=(15, 10), sharex=True, sharey=True)

    # Plotting full data
    axes[0].set_title("Full Data - Location")
    full_gdf.plot(ax=axes[0], color='blue', markersize=5, alpha=0.5,
                  label='Pings')

    # Highlight the home location
    home_point = gpd.GeoDataFrame(
        [{'geometry': gpd.points_from_xy([full_loc['lng']],
                                         [full_loc['lat']])[0]}],
        crs='EPSG:4326'
    )

    sns.kdeplot(x=full_gdf.geometry.x, y=full_gdf.geometry.y,
                ax=axes[0], cmap="Blues", fill=True, alpha=0.7)

    home_point.plot(ax=axes[0], color='red', markersize=40, label='Home '
                                                                  'Location')
    ctx.add_basemap(ax=axes[0], crs=full_gdf.crs.to_string(),
                    source=ctx.providers.OpenStreetMap.Mapnik)
    # Add the address below the plot
    axes[0].text(0.5, 0.05, full_loc['address'], ha='center', va='bottom',
                 fontsize=12, transform=axes[0].transAxes)

    home_point_reduced = gpd.GeoDataFrame(
        [{'geometry': gpd.points_from_xy([reduced_loc['lng']],
                                         [reduced_loc['lat']])[0]}],
        crs='EPSG:4326'
    )
    # Plotting reduced data home locations
    axes[1].set_title("Home Location")
    reduced_gdf.plot(ax=axes[1], color='blue', markersize=5, alpha=0.5,
                     label='Pings')
    sns.kdeplot(x=reduced_gdf.geometry.x, y=reduced_gdf.geometry.y,
                ax=axes[1], cmap="Blues", fill=True, alpha=0.7)
    home_point_reduced.plot(ax=axes[1], color='red', markersize=40,
                            label='Home Location')
    ctx.add_basemap(ax=axes[1], crs=reduced_gdf.crs.to_string(),
                    source=ctx.providers.OpenStreetMap.Mapnik)
    # Add the address below the plot
    axes[1].text(0.5, 0.05, reduced_loc['address'], ha='center', va='bottom',
                 fontsize=12, transform=axes[1].transAxes)

    fig.suptitle(f'Home location plots for user '
                 f'{full_df.iloc[0].user_id}', fontsize=16)
    # Setting shared axis properties
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def find_home_locations(df, user_ids, eps_km=100, min_samples=10):
    """
    Find the home locations for a list of users using DBSCAN.

    Parameters:
        df (pd.DataFrame): DataFrame containing user pings with columns [
        'user_id', 'latitude', 'longitude', 'timestamp'].
        user_ids (list): List of user IDs to find home locations for.
        eps_km (int): The radius in kilometers to consider for the DBSCAN
        neighborhood. Default is 100 meters.
        min_samples (int): Minimum number of samples in a neighborhood for
        DBSCAN. Default is 10.

    Returns:
        dict: A dictionary with user IDs as keys and their home locations as
        {latitude, longitude, address_name}.
    """
    home_locations = {}
    kms_per_radian = 6371.0088
    eps = eps_km / 1000 / kms_per_radian  # Convert epsilon to radians

    for user_id in user_ids:
        night_ping = df[df['user_id'] == user_id].copy()

        if night_ping.empty:
            print(f"No data for user {user_id}")
            continue

        # Convert coordinates to radians
        coords = night_ping[['latitude', 'longitude']].apply(np.radians)

        db = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree',
                    metric='haversine').fit(coords)

        night_ping['cluster'] = db.labels_

        # Find the cluster with the most pings
        home_cluster = night_ping['cluster'].value_counts().idxmax()
        if home_cluster == -1:
            print(f"No significant cluster for user {user_id}")
            continue

        # Get the average (mean) coordinates of the home cluster
        home_location = night_ping[night_ping['cluster'] == home_cluster][[
            'latitude', 'longitude']].mean()
        home_locations[user_id] = {
            'lat': home_location['latitude'],
            'lng': home_location['longitude'],
            'address': get_address_from_coord(home_location['latitude'],
                                              home_location['longitude'])
        }

    return home_locations


sample_users = [46, 114, 169]
full_data_loc = find_home_locations(night_pings, sample_users)

# Example of sampling 10% of the data
reduced_data = night_pings.groupby('user_id').apply(lambda x: x.sample(
    frac=0.1))

reduced_data_loc = find_home_locations(reduced_data, sample_users)

# Plotting a comparison plot for a single user
comparing_user = 169

# plot_full_vs_reduced_data(
#     night_pings[night_pings['user_id'] == comparing_user],
#     full_data_loc[comparing_user],
#     reduced_data[reduced_data['user_id'] == comparing_user],
#     reduced_data_loc[comparing_user]
# )


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
plot_user = 163
user_loc = find_home_locations(night_pings, [plot_user])
plot_user_home_pings(
    night_pings[night_pings['user_id'] == plot_user],
    user_loc[plot_user]
)

