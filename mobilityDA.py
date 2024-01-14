import pandas as pd
import glob
import os
import osmnx as ox
from shapely.geometry import Point
from tqdm import tqdm

# Global Parameters
bar_length = 120

base_dir = "./Data"
place_name = "Beijing, China"
place_boundary = ox.geocode_to_gdf(place_name)
start_date = "2010-01-01"
end_date = "2010-06-30"
raw_csv_path = "./raw_trajectory_df.csv"
filtered_csv_path = "./filter_trajectory_df.csv"


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
raw_df = read_all_plt()

if os.path.exists(filtered_csv_path):
    print(f"CSV file found at {filtered_csv_path}. Reading the DataFrame "
          f"from CSV.")
    filter_df = filter_users_by_min_pings(load_csv(filtered_csv_path), 1000)
    save_csv(filter_df, filtered_csv_path)
else:
    print("Start filtering dataframe by region and date")
    filter_df = filter_by_region(filter_by_date(raw_df, start_date, end_date))
    save_csv(filter_df, filtered_csv_path)


# Next Step:
# 1. Create a github repo for this project
# 2. Discard users with very few pings in that preiod in the specified location
# 3. Describe the methodology and tools used to prepare this data and
# provide some insights on its completeness and density
# 4. Parse and transform the data to be able to identify pings that occurred
#  at night (you are free to interpret what "night" means, as different
# ranges could be valid or convenient
# 5. Plot the distribution of these pings for some users.
# 6. How do they compare with those during the day?
# 7. Propose some algorithm of your preference for clustering these
# nighttime pings with the goal of identifying home locations and
# distinguishing them from other locations.
# 8. Describe this algorithm in detail and how it will tackle problems like
# data sparsity, people staying in hotels or other residences many nights,
# and people moving to a different location permanently. What is the
# complexity of this algorithm?
# 9. Implement this algorithm for a few users and provide the necessary
# code. Would this algorithm work the same if the users had 10% of the data
# they have? Compare the results.
# 10. Would it be challenging to scale this to multiple users? Provide a few
#  plots to support your reasoning about your findings.


