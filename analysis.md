# GPS Mobility Data Analysis
## Data Pre-processing
### Data Loading and Parsing
Python is used for scripting and data manipulation. Pandas: A Python 
library for data analysis, used to load .plt files into DataFrame 
structures for easy data manipulation. Scripts are written to iterate 
through the .plt files, parsing the GPS data (latitude, longitude, and 
timestamp) and associating it with respective user IDs.

### Data Transformation
- Implement a function to filter the DataFrame based on a date range. This 
  function should keep only the rows where the timestamp is within the 
  specified start and end dates.
- If focusing on Beijing, use geospatial tools like GeoPandas to filter out 
  pings outside the Beijing area. This might involve spatial joins with a 
  shapefile of Beijing.
- Group data by user ID and count the number of pings per user. Retain only 
  those users who have a ping count above a certain threshold, indicating active or frequent use. This step is crucial for robust analysis, as users with very few pings might not provide meaningful insights.

### Data Storage and UI
Raw and processed DataFrame are stored in a Comma-Separated Values (CSV) 
format. This format is widely used due to its simplicity and compatibility 
with various data analysis tools and platforms. Hence, we are able to reuse 
these data easily instead of reading all the .plt data again during 
development.

Process bar will be shown in the console during some timing task such as 
recursively reading all the .plt files so that the progress can be shown 
visually.

### Insights on Completeness and Density
**Completeness**

The dataset seems relatively complete for certain users (e.g., user IDs 128 
and 153 with ping counts of 176,240 and 203,334, respectively), indicating 
extensive tracking. However, there are users with very low ping counts (e.g.
, user ID 178 with only 84 pings), suggesting sporadic data collection or 
limited user participation.

**Density**

The dataset exhibits high variability in density. Some users have dense 
data points, implying frequent use or longer tracking periods, which are 
beneficial for detailed analysis. Conversely, the lower density for some 
users might limit the analysis scope, particularly for studies requiring 
consistent temporal coverage or frequent spatial points. The density 
variation raises questions about the dataset's representativeness. Higher 
density in some users' data might bias analyses towards areas or in which 
these users were active.

The variation in data completeness and density necessitates careful data 
filtering. Analyses might need to segment users into categories based on 
activity levels.

## Data Analysis
### Identify nighttime locations 

**Description of Plotting Method**

To visualize the distribution of day and night pings for selected users, 
I have created a series of plots that include both individual pings and their 
density heatmaps. This can be done using a combination of geopandas for 
geospatial data handling, matplotlib for plotting, and seaborn for density 
heatmaps. The visualization consists of two rows of subplots: the first row 
shows individual pings on a map, and the second row displays the 
corresponding density heatmaps.

**Analysis of Night vs. Day Pings**

During the day, the concentration of pings primarily within the city zone 
suggests a strong focus on urban centers, likely due to work, education, 
and daily errands, which are typically centered in more densely populated 
areas. This pattern aligns with typical daytime activities, where 
individuals converge in commercial, industrial, or central urban areas. In 
contrast, the night plot showing a wider spread indicates a dispersal from 
these concentrated zones to more varied locations, possibly reflecting a 
return to residential areas that are more spread out, or engagement in 
social and recreational activities that take place in different, perhaps 
less central, parts of the city. This shift can highlight the transition 
from professional and structured daytime activities to more personal and 
leisure-oriented nighttime activities. Additionally, the expanded spread at 
night might also suggest different lifestyle patterns, such as visiting 
friends or family, dining out, or attending cultural events, which often 
occur in diverse locations beyond the core urban area.

Besides, the observed shift in activity centers and movement patterns 
between day and night vividly showcases the dynamic nature of urban life. 
Daytime is marked by concentrated activity in economic and commercial hubs, 
with dense pings and distinct movement paths indicating active commuting 
and work-related activities. In contrast, nighttime shows a dispersion of 
activities, with a move towards residential areas or scattered 
entertainment spots. This change reflects a transition from the bustling, 
transit-oriented daytime to a more localized and leisure-focused night, 
highlighting the dual roles urban spaces play in accommodating work and 
leisure, a key insight for urban planning and service optimization.

### Identify home locations

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is an 
effective algorithm for clustering nighttime pings to identify home 
locations. DBSCAN excels in identifying clusters of arbitrary shapes and 
sizes, making it well-suited for geospatial data that can be irregularly 
distributed. It operates by grouping points that are closely packed 
together and marking points as outliers if they lie alone in low-density 
regions. This characteristic is particularly beneficial for distinguishing 
permanent residences from transient stays like hotels, as the latter would 
typically not exhibit the same level of nightly recurrence in the data.

For individuals staying at multiple residences or those who have moved 
permanently, DBSCAN's flexibility in forming clusters based on density 
allows for the identification of multiple potential home locations. This 
adaptability, however, means the algorithm might need supplementary 
temporal analysis to discern between past and current residences.

By adjusting its eps (maximum distance between two samples) and min_samples 
(minimum points to form a dense region) parameters, DBSCAN effectively 
handles data sparsity and addresses challenges like multiple residences or 
relocation. Its performance, typically O(n log n), makes it suitable for 
large datasets, though its efficiency depends on the choice of eps and data 
quality. This approach requires careful preprocessing to manage GPS data 
inaccuracies and ensure reliable identification of home locations. By using 
the pings during night, it can more accurately determine the home location 
instead of getting some locations like work spaces etc. since users will 
probably stay around home location during the night.

**Reducing Data to 10%**

When the data is reduced to 10%, the effectiveness of clustering algorithms 
like DBSCAN in identifying home locations can be significantly impacted, 
leading to a marked difference in results compared to using the full 
dataset. With only a tenth of the original data, key patterns necessary for 
accurate clustering might be lost, resulting in a failure to form 
meaningful clusters or misidentification of home locations due to the 
change in density dynamics. The reduction in data points increases sparsity,
disrupting the density-dependent clustering process that DBSCAN relies on. 
This is especially pronounced for users with originally sparse or irregular 
pings, where the reduced data may no longer represent their typical 
movement patterns or frequent locations. 

Consequently, the algorithm may either fail to identify a cluster that 
confidently represents a home location or may erroneously designate a less 
frequently visited area as the home due to the randomness of the data 
reduction. Adjusting the parameters of DBSCAN, such as eps (the 
neighborhood size) and min_samples (the minimum points to form a dense 
region), could partly mitigate these issues. However, the intrinsic loss of 
data continuity and density often requires a more nuanced approach or 
alternative methods to maintain the accuracy of home location 
identification in such significantly reduced datasets.

**Scaling to multiple users**

Scaling the process of identifying home locations using clustering 
algorithms like DBSCAN to multiple users can indeed present several challenges:

1. Variability in User Patterns:
  Different users may have distinct patterns in terms of their mobility, 
   frequency of pings, and areas visited. A single set of DBSCAN parameters 
   (eps and min_samples) might not be optimal for all users, requiring 
   individual parameter tuning.

2. Data Sparsity and Density: For users with fewer pings, the algorithm 
   might struggle to form meaningful clusters. Conversely, for users with 
   dense data, the algorithm might create too many clusters, complicating the 
   identification of the actual home location.

3. Computational Resources: Processing a large number of users, each with 
   potentially thousands of location pings, requires significant 
   computational resources. The complexity increases with the volume of 
   data and the need for parameter optimization for each user.