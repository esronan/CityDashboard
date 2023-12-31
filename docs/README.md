# What is this project about?
Inspired by the city dashboard project by Maynooth University, I decided to create a city dashboard for London. The intention here is creating an accessible, interactive and flexible dashboard that provides an overview over a city for interested citizens, urban planners, and sociological researchers. The potential variables to include are limitless, so I have restricted myself here to data on housing prices, rental prices, airbnb rental prices, income levels and satisfaction levels.

# How does it work?
The project adopted a simple ETL pipeline. The data was extracted from the respective sources, transformed into a comprehensive datatable, and then loaded onto a Google Cloud instance, to be queried using SQL later. The dashboard itself was built in Streamlit, to allow for quick development of ideas in a Python-native environment.

Data sources: The majority was retrieved from London DataStore, an open data-sharing portal hosting data from the Greater London Authority, Airbnb data was available on Kaggle, and borough geographical information was scraped from wikipedia. 

Data transformation: Monthly data on variables was aggregated to merge with yearly data. Easting/Northing data was converted into latitude and longitude. Data was formatted to allow for table joins. New features were engineered such as "distance to centre", "income to house price ratio", and "population density". 

# Where from here?
The intention is to build the app to cover a greater range of variables, include some thematic investgations and scrape certain data sources for up-to date information. Variables of interest could be air quality, traffic levels, income, demographic data, unemployment rates, and otherwise. Thematic analysis could include gentrification, the housing market, liveability, crime, and multiculturality.

# Images
### Home page
![Home](../images/image-1.png)
### Airbnb listings page
![Airbnb listings](../images/image-5.png)
### Data explorer page
![Data explorer](../images/image-2.png)
![Data explorer 2](../images/image-3.png)
![Data explorer 3](../images/image-4.png)