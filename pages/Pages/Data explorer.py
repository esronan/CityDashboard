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
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/Administrator/Documents/GitHub/StreamlitApp/data/lyrical-edition-403712-0575fe01e220.json"

st.set_page_config(layout="centered")

st.set_option('deprecation.showPyplotGlobalUse', False) #Turn off deprecation warnings for heatmap visualisation

#plotly use_container_width=True

st.title('London housing data')
st.write("A tool allowing you to compare house prices across london boroughs in combination with a variety of other relevant factors")



##### LOAD DATA ######

#initiate connector
connector = Connector()

#initiate connection engine
engine = utils.init_connection_engine(db_instance=db_instance, db_user=db_user, db_pass=db_pass, db_name=db_name, connector_instance=connector)

try:
    query = "SELECT * FROM processed_data"
    merged_data = pd.read_sql(query, con=engine)
except:
    merged_data = pd.read_csv("C:/Users/Administrator/Documents/GitHub/StreamlitApp/data/processed_data.csv")
    st.write("Please let home tab load first.")

if st.checkbox('Show processed data'):
    st.subheader('Processed data')
    st.write(merged_data)


##### DATA EXPLORATION #####

#pr = df.profile_report()
#st_profile_report(pr)

# Define coloums of interest for widgets
col_of_int = ['median_salary', 'mean_salary', 'income_price_ratio', 'life_satisfaction',
       'recycling_pct', 'population_size', 'population_density', 'number_of_jobs',    
       'area_size', 'no_of_houses', 'average_price', 'distance_to_centre']

def multiselect_widget(data, col, var_name):
    #Widget for user to choose areas to compare
    vals = data[col].values
    chosen_vals = st.multiselect(f"Pick {var_name} to compare", vals)
    filtered_data = data[data[col].isin(chosen_vals)]
    return filtered_data

filtered_data = multiselect_widget(merged_data, "area", "areas")

#Widget for user to choose variable to compare on
y = st.selectbox("Pick a variable to compare on", col_of_int)

#Pivot table to get time series data for chosen variable
data_pivot = filtered_data.pivot(index='date', columns='area', values=y)

st.subheader("Compare boroughs over time")
#graph
x = "date"
st.line_chart(data=data_pivot)

st.subheader("Relationship between variables")
#heatmap

utils.create_heatmap(merged_data, col_of_int)

#
st.write("As can be seen here, a few variables like population size, number of jobs, area size and number of houses are understandably all very highly correlated. Mean salary and median salary also have a high correlation, though not as high - their difference gives important information on the spread of incomes (and hence inequality).")



##### MAP #####
st.subheader("Visualising variation on a map")
map_col = st.selectbox("Pick a variable", col_of_int)
year = st.selectbox("Pick a year", merged_data["date"].unique())

merged_data = utils.get_boroughs(merged_data)
type_grad = "size"
df = utils.scale_to_max(merged_data, map_col, type=type_grad)
df = df[df["date"] == year]
df = df[df["area"] != "city of london"]
#st.write(df)
st.map(df, size=type_grad)

if st.button('Source'):
     st.write('Data drawn from Justinas Cirtautas\' dataset "Housing in London", in turn drawing from London Datastore, as well as the Wikipedia page on London boroughs.\n\nLinks: \nhttps://www.kaggle.com/datasets/justinas/housing-in-london\nhttps://en.wikipedia.org/wiki/List_of_London_boroughs')
else: 
    ""

    