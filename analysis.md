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