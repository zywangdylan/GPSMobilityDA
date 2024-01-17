# Home Location Identifier from Geospatial Data
This repository contains code and instructions for a project aimed at identifying potential home locations from geospatial data. The project utilizes the DBSCAN clustering algorithm to analyze user location data, primarily focusing on nighttime pings to determine common stay points that could represent users' home locations.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
Before you begin, ensure you have the following installed:

- Python 3.x
- Required Python libraries: pandas, scikit-learn, geopandas, matplotlib, contextily
- (Optional) Jupyter Notebook, if you prefer to run the code in an interactive environment

You can install the necessary Python libraries using pip:
```
pip install pandas numpy glob2 osmnx geopandas contextily matplotlib seaborn tqdm datetime geopy scikit-learn
```
This command installs the following packages:
- pandas: For data manipulation and analysis.
- numpy: For numerical computing.
- glob2: For file path pattern matching.
- osmnx: For retrieving, constructing, analyzing, and visualizing street networks.
- geopandas: For working with geospatial data.
- contextily: For adding basemaps to plots.
- matplotlib: For creating static, interactive, and animated visualizations.
- seaborn: For data visualization based on matplotlib.
- tqdm: For progress bars.
- datetime: For manipulating dates and times.
- geopy: For geocoding addresses.
- scikit-learn: For machine learning and data mining.

## Data Setup
The raw data can be downloaded [here](https://www.microsoft.com/en-us/download/details.aspx?id=52367). This dataset was prepared as part of the research publication [1] Yu Zheng, Lizhu Zhang, Xing Xie, Wei-Ying Ma. Mining interesting locations and travel sequences from GPS trajectories. In Proceedings of International conference on World Wild Web (WWW 2009), Madrid Spain. ACM Press: 791-800.

You can also directly download preprocessed data frames in csv format from [here](https://drive.google.com/drive/folders/14sMbGNylIkKfwCsDkmwCqOYqIfUYDogD?usp=sharing) which includes:
- `day_trajectory_df.csv`: Contains daytime trajectory data.
- `filter_trajectory_df.csv`: Contains filtered trajectory data (Beijing in the first 6 months of 2010).
- `night_trajectory_df.csv`: Contains nighttime trajectory data.
- `raw_trajectory_df.csv`: Contains the raw trajectory data.

## Usage
The main functionality is contained in `mobilityDA.py`, which includes data loading, preprocessing, DBSCAN clustering, and visualization of the results. Run the script/notebook to see the output plots and home location analysis. You can comment/uncomment some codes for plotting to see the plots you want. You can find the analysis in the `analysis.md` file.
















