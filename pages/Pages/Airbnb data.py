import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
#from streamlit_pandas_profiling import st_profile_report
from geopy.distance import geodesic
from sqlalchemy import create_engine, inspect
from google.cloud.sql.connector import Connector, IPTypes
import pymysql
import os
import sys
sys.path.append('C:/Users/Administrator/Documents/GitHub/StreamlitApp/data')
sys.path.append('C:/Users/Administrator/Documents/GitHub/StreamlitApp/utils')
from keys import db_instance, db_user, db_pass, db_name
import utils
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/Administrator/Documents/GitHub/StreamlitApp/data/lyrical-edition-403712-0575fe01e220.json"
####
st.title("Airbnb listings data")
st.write("Here you can visualise publically available Airbnb data in London from the year of 2022.")

connector = Connector() 

airbnb_data = pd.read_csv("C:/Users/Administrator/Documents/GitHub/StreamlitApp/data/airbnb_listings.csv")

filtered_data = utils.multiselect_widget(airbnb_data, "neighbourhood", "neighbourhoods")
filtered_data = utils.scale_to_max(filtered_data, col="price", type="colour")
#print(airbnb_data["colour"].unique(), airbnb_data["price"].describe(), airbnb_data["price"].median())
st.map(filtered_data, color='colour', size=10)