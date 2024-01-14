
# GPS Mobility Data Analysis Take Home Task for Research Assistants

Welcome to the **GPS Mobility Data Analysis** take home task for research assistant candidates at the CSSLab. This task is intended to test your coding, data analysis, and visualization skills and will be taken into account when considering your application. We estimate that this task should take approximately four hours to complete, but you can have a whole week if needed. 

The goal of this task is to analyze and visualize visits data from **GeoLife**, which is available on an online repository and is free to use. The data has been stored and organized per user in .plt files. This project will require you to load, parse and analyze this type of data to investigate home locations for users in the city of Beijing in 2010. Your results should be presented in a short report emailed to shape@seas.upenn.edu with all code and graphics used for producing them also attached (you can choose any programming language that you prefer). 

## 1. Data Ingestion and Processing 
Download data from the dataset's Github repository and process it into a format suitable for analysis. The data can be downloaded [here](https://www.microsoft.com/en-us/download/details.aspx?id=52367), where you will also find all the appropriate documentation. This dataset was prepared as part of the research publication [1] Yu Zheng, Lizhu Zhang, Xing Xie, Wei-Ying Ma. Mining interesting locations and travel sequences from GPS trajectories. In Proceedings of International conference on World Wild Web (WWW 2009), Madrid Spain. ACM Press: 791-800. You should subset this data to pings in the city of Beijing in 2010---this will require finding a shapefile or geopandas file with the city geometry. You should subset the data to the first 6 months of 2010 and you are encouraged to discard users with very few pings in that preiod in the specified location. Please describe the methodology and tools used to prepare this data and provide some insights on its completeness and density. 

## 2. Identify nighttime locations as candidate home locations
Parse and transform the data to be able to identify pings that occurred at night (you are free to interpret what "night" means, as different ranges could be valid or convenient. Can you plot the distribution of these pings for some users? How do they compare with those during the day?

## 4 Propose a way to identify home locations
Propose some algorithm of your preference for clustering these nighttime pings with the goal of identifying home locations and distinguishing them from other locations. You are encouraged to do a brief search in the literature for this. Describe this algorithm in detail and how it will tackle problems like data sparsity, people staying in hotels or other residences many nights, and people moving to a different location permanently. What is the complexity of this algorithm?

## 5 Implementation
Implement this algorithm for a few users and provide the necessary code. Would this algorithm work the same if the users had 10% of the data they have? Compare the results. Would it be challenging to scale this to multiple users? Provide a few plots to support your reasoning about your findings. 

## Conclusion 
We look forward to reviewing your work! Please don’t hesitate to reach out if you have any questions or need help understanding any part of this task - we are here to help!
