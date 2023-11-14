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
import plotly.express as px


sys.path.append('../data')
sys.path.append('../utils')
from keys import db_instance, db_user, db_pass, db_name
import utils

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/Administrator/Documents/GitHub/StreamlitApp/data/lyrical-edition-403712-0575fe01e220.json"

st.set_option('deprecation.showPyplotGlobalUse', False) #Turn off deprecation warnings for heatmap visualisation

st.set_page_config(layout="wide")

st.title('London housing data')

#initiate connector
connector = Connector()

#initiate connection engine
engine = utils.init_connection_engine(db_instance=db_instance, db_user=db_user, db_pass=db_pass, db_name=db_name, connector_instance=connector)

merged_data = utils.load_processed_data()

##### DASHBOARD #####

#Average price in borough, London, UK (+ over time), average salary



avg_price_now = utils.get_var(merged_data, "average_price", "london", 2019)
avg_price_then = utils.get_var(merged_data, "average_price", "london", 1999)

avg_sal_now = utils.get_var(merged_data, "median_salary", "london", 2019)
avg_sal_then = utils.get_var(merged_data, "median_salary", "london", 1999)

cheapest_borough = utils.get_minmax(merged_data, "average_price", 2019, "min")
dearest_borough = utils.get_minmax(merged_data, "average_price", 2019, "max")


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

