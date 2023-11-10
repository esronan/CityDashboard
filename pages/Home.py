import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
#from streamlit_pandas_profiling import st_profile_report
import requests
import re
from geopy.distance import geodesic
import pandas as pd
from sqlalchemy import create_engine, inspect
from google.cloud.sql.connector import Connector, IPTypes
import pymysql
import os
import sys
sys.path.append('../data')
sys.path.append('../utils')
from keys import db_instance, db_user, db_pass, db_name
import utils
import plotly.express as px
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/Administrator/Documents/GitHub/StreamlitApp/data/lyrical-edition-403712-0575fe01e220.json"

st.set_option('deprecation.showPyplotGlobalUse', False) #Turn off deprecation warnings for heatmap visualisation

st.set_page_config(layout="wide")

st.title('London housing data')

#initiate connector
connector = Connector()

#initiate connection engine
engine = utils.init_connection_engine(db_instance=db_instance, db_user=db_user, db_pass=db_pass, db_name=db_name, connector_instance=connector)

#query from SQL database. If not, load raw data and process
try:
    query = "SELECT * FROM processed_data"
    merged_data = pd.read_sql(query, con=engine)
except:
    st.write("Processing data...")

    ##### LOAD DATA ######

    yearly_data_url = "C:/Users/Administrator/Documents/GitHub/StreamlitApp/data/housing_in_london_yearly_variables.csv"
    monthly_data_url = "C:/Users/Administrator/Documents/GitHub/StreamlitApp/data/housing_in_london_monthly_variables.csv"
    
    #st.subheader("Loading Data")
    data_load_state = st.text('Loading data...')
    data = utils.load_data(yearly_data_url)
    data_monthly = utils.load_data(monthly_data_url)
    data_load_state.text("Data loaded!")



    ##### PROCESS DATA ######

    data = utils.process_yearly_data(data)
    data_monthly = utils.agg_monthly_data(data_monthly)


    #GEO DATA

    #scrape borough coords
    url = "https://en.wikipedia.org/wiki/List_of_London_boroughs" 
    df = utils.get_borough_coords(url)

    #Convert latitude and longitude to positive/negative decimals
    df['latitude'] = df['latitude'].apply(utils.convert_coords, type = "lat")
    df['longitude'] = df['longitude'].apply(utils.convert_coords, type="lon")

    #Get distance to centre
    df["distance_to_centre"] = df.apply(lambda row: utils.dist_centre(row['latitude'], row['longitude']), axis=1)
    df["distance_to_centre"] = df["distance_to_centre"].astype(float)
    df = df[["Borough", "latitude", "longitude", "distance_to_centre"]]

    #Format borough names
    df = utils.format_borough_names(df)

    #join
    merged_data = data.merge(data_monthly, on=['area', 'date'], how='left')
    merged_data = merged_data.merge(df, on="area", how='left')

    #Create income_price ratio
    merged_data["income_price_ratio"] = merged_data["average_price"]/merged_data["mean_salary"]


    ##### SQL UPLOAD #####

    #initiate connector#
    connector = Connector() 

    #initiate connection engine
    engine = utils.init_connection_engine(db_instance=db_instance, db_user=db_user, db_pass=db_pass, db_name=db_name, connector_instance=connector)

    #save to database
    merged_data.to_sql('processed_data', con=engine, if_exists='replace', index=False)

    #save as CSV
    merged_data.to_csv("../data/processed_data.csv")

##### DASHBOARD #####

#Average price in borough, London, UK (+ over time), average salary
def get_var(data, col, area, year):
    mask = (data["area"]== area) & (data["date"] == year)
    val = data[mask][col].iloc[0]
    return val

def get_minmax(data, col, year, agg):
    data = utils.get_boroughs(data)
    mask = (data["date"] == year)
    if agg == "min":
        minval = data[mask][col].argmin()
        area = data.iloc[minval]["area"]
    elif agg == "max":
        minval = data[mask][col].argmax()
        area = data.iloc[minval]["area"]
    return area



avg_price_now = get_var(merged_data, "average_price", "london", 2019)
avg_price_then = get_var(merged_data, "average_price", "london", 1999)

avg_sal_now = get_var(merged_data, "median_salary", "london", 2019)
avg_sal_then = get_var(merged_data, "median_salary", "london", 1999)

cheapest_borough = get_minmax(merged_data, "average_price", 2019, "min")
dearest_borough = get_minmax(merged_data, "average_price", 2019, "max")


kpi1, kpi2, kpi3 = st.columns(3)
 

kpi1.metric(
    label="Average property price",
    value="£" + str(round(avg_price_now)),
    delta="£" + str(round(avg_price_now)-round(avg_price_then)), 
    delta_color="off"
)

kpi2.metric(
    label="Median salary",
    value="£" + str(int(avg_sal_now)),
    delta="£"  + str(int(avg_sal_now-avg_sal_then)),
    delta_color="off"
)
kpi3.metric(
    label="Cheapest borough (and most expensive)",
    value=cheapest_borough.title(),
    delta=dearest_borough.title(),
    delta_color = "off"
)


fig_col1, fig_col2 = st.columns(2)
with fig_col1:
    st.markdown("### Average property price over time")
    filtered_data = merged_data[merged_data["area"].isin(["london", "england", "north west", "north east", "south west", "south east", "east midlands", "west midlands", "east", "yorkshire and the humber"])]
    #Pivot table to get time series data for chosen variable
    data_pivot = filtered_data.pivot(index='date', columns='area', values="average_price")
    fig = px.line(data_pivot)
    # fig = px.density_heatmap(
    #     data_frame=df, y="age_new", x="marital")
    st.write(fig)

with fig_col2:
    st.markdown("### Average property price per borough over time")
    boroughs = utils.get_boroughs(merged_data)
    fig = px.density_heatmap(
        data_frame=boroughs, x="date", y="area", z="average_price"
    )
    st.write(fig)


merged_data["income_price_ratio"] = merged_data["average_price"]/merged_data["mean_salary"]
filtered_data = merged_data[merged_data["area"].isin(["london", "england", "north west", "north east", "south west", "south east", "east midlands", "west midlands", "east", "yorkshire and the humber"])]
#Pivot table to get time series data for chosen variable
data_pivot = filtered_data.pivot(index='date', columns='area', values="income_price_ratio")
fig = px.line(data_pivot)
#st.write(fig)

