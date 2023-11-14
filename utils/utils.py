import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import re
from geopy.distance import geodesic
import pandas as pd
from sqlalchemy import create_engine, inspect
from google.cloud.sql.connector import Connector, IPTypes
import pymysql
import sys

sys.path.append('../data')
from keys import db_instance, db_user, db_pass, db_name

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

def process_yearly_data(data): 
    data["area"] = data["area"].astype("string")
    data["mean_salary"] .replace('-', np.nan, inplace=True)
    data["mean_salary"] .replace('#', np.nan, inplace=True)
    data["mean_salary"] = data["mean_salary"].astype(float)
    data["median_salary"] = data["median_salary"].astype(float, errors="ignore")
    data["life_satisfaction"] = data["life_satisfaction"].astype(float, errors="ignore")
    data["recycling_pct"] .replace('na', np.nan, inplace=True)
    data["recycling_pct"] = data["recycling_pct"].astype(float)
    data['date'] = pd.to_datetime(data['date']).dt.year
    data['population_density'] = data['population_size']/data['area_size']
    return data

def agg_monthly_data(data):
    #Extract year
    data["date"] = pd.to_datetime(data["date"]).dt.year
    #Aggregate data to prepare for merging with 
    data = data.groupby(["area", "date"]).agg({'average_price': 'mean', 'houses_sold': 'sum', 'no_of_crimes': 'sum', 'code': 'max'})
    return data

def get_borough_coords(url):
    response = requests.get(url)
    html_content = response.text
    df = pd.read_html(html_content)[0]
    df['Co-ordinates'] = df['Co-ordinates'].str.split('/', expand=True)[1]
    df[['latitude', 'longitude']] = df['Co-ordinates'].str.split(' ', expand=True).iloc[:,1:]
    return df


def convert_coords(coord, type="lat"):
    #remove unicode
    coord = coord.replace('\ufeff', '')
    if type == "lat":
        coord = float(coord[:-2]) * (1 if coord[-1].upper() == 'N' else -1)
    elif type == "lon":
        coord = float(coord[:-2]) * (1 if coord[-1].upper() == 'E' else -1)
    else:
        st.write("FAIL")
    return coord

#Calculate distance to centre of city using Haversin formula
def dist_centre(latitude, longitude):
    centre = (51.49722222222222,-0.13722222222222222) #Coordinates for Westminster borough
    return geodesic(centre,(latitude, longitude)).km


def format_borough_names(df):
    df['Borough'] = df['Borough'].str.extract(r'^(.*?)(?:\[|$)')
    df['Borough'] = df['Borough'].apply(lambda x: " ".join([a.lower() for a in x.split()]))
    df = df.rename(columns={"Borough":"area"})
    return df

def get_table_names(engine):
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(tables)

def create_heatmap(df, columns):
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[columns].corr()
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='viridis', vmin=-1, vmax=1)
    st.pyplot()

def scale_to_max(df, col, type="size"):
    if type == "size":
        df["size"] = 3000*df[col]/df[col].max()
    elif type == "colour":
        #cap max at 2000
        df['colour'] = df["price"].apply(lambda x: min(x, 500))
        #min max scale
        df['colour'] = (df['colour'] - df['colour'].mean()) / df['colour'].std()
        #df['colour'] = (df['colour'] - df['colour'].min())  / (df['colour'].max() - df['colour'].min())
        df["colour"] = df["colour"].apply(lambda x: (35*x, abs(35*(1-x)),0))
    else:
        print("Choose a valid type!")
    return df

def min_max_scaler(df, col, type="size"):
    
    df["size"] = df[col]-df[col].min()/(df[col].max()-df[col].min())
    if type=="colour":
        df["colour"] = df["size"].apply(lambda x: (255*x, abs(255*(1-x)),0))
    return df
# Standalone getconn function
def getconn(db_instance, db_user, db_pass, db_name, connector_instance):
    conn = connector_instance.connect(
        db_instance,
        "pymysql",
        user=db_user,
        password=db_pass,
        db=db_name,
        ip_type=IPTypes.PUBLIC,
    )
    return conn

# The init_connection_engine function, adjusted to use the standalone getconn function
def init_connection_engine(db_instance, db_user, db_pass, db_name, connector_instance):
    engine = create_engine("mysql+pymysql://", creator=lambda: getconn(db_instance, db_user, db_pass, db_name, connector_instance))
    return engine


def get_boroughs(df):
    return df[df["borough_flag"] == 1]

def multiselect_widget(data, col, var_name):
    #Widget for user to choose areas to compare
    vals = list(data[col].unique()) + [list(data[col].unique())]
    chosen_vals = st.multiselect(f"Pick {var_name} to compare", vals)
    filtered_data = data[data[col].isin(chosen_vals)]
    return filtered_data

def load_processed_data():
    try:
        query = "SELECT * FROM processed_data"
        merged_data = pd.read_sql(query, con=engine)
        return merged_data
    except:
       # st.write("Processing data...")

        ##### LOAD DATA ######

        yearly_data_url = "C:/Users/Administrator/Documents/GitHub/StreamlitApp/data/housing_in_london_yearly_variables.csv"
        monthly_data_url = "C:/Users/Administrator/Documents/GitHub/StreamlitApp/data/housing_in_london_monthly_variables.csv"
        
        #st.subheader("Loading Data")
        #data_load_state = st.text('Loading data...')
        data = load_data(yearly_data_url)
        data_monthly = load_data(monthly_data_url)
       # data_load_state.text("Data loaded!")

        #scrape borough coords
        url = "https://en.wikipedia.org/wiki/List_of_London_boroughs" 
        df = get_borough_coords(url)

        ##### PROCESS DATA ######

        data = process_yearly_data(data)
        data_monthly = agg_monthly_data(data_monthly)


        #GEO DATA

        #Convert latitude and longitude to positive/negative decimals
        df['latitude'] = df['latitude'].apply(convert_coords, type = "lat")
        df['longitude'] = df['longitude'].apply(convert_coords, type="lon")

        #Get distance to centre
        df["distance_to_centre"] = df.apply(lambda row: dist_centre(row['latitude'], row['longitude']), axis=1)
        df["distance_to_centre"] = df["distance_to_centre"].astype(float)
        df = df[["Borough", "latitude", "longitude", "distance_to_centre"]]

        #Format borough names
        df = format_borough_names(df)

        #join
        merged_data = data.merge(data_monthly, on=['area', 'date'], how='left')
        merged_data = merged_data.merge(df, on="area", how='left')

        #Create income_price ratio
        merged_data["income_price_ratio"] = merged_data["average_price"]/merged_data["mean_salary"]


        ##### UPLOAD PROCESSED DATA TO SQL INSTANCE #####

        #initiate connector#
        connector = Connector() 

        #initiate connection engine
        engine = init_connection_engine(db_instance=db_instance, db_user=db_user, db_pass=db_pass, db_name=db_name, connector_instance=connector)

        #save to database
        try:
            merged_data.to_sql('processed_data', con=engine, if_exists='replace', index=False)
        except:
            print("SQL instance unavailable")

        #save as CSV
        merged_data.to_csv("../data/processed_data.csv")
        return merged_data

def get_var(data, col, area, year):
    mask = (data["area"]== area) & (data["date"] == year)
    val = data[mask][col].iloc[0]
    return val

def get_minmax(data, col, year, agg):
    data = get_boroughs(data)
    mask = (data["date"] == year)
    if agg == "min":
        minval = data[mask][col].argmin()
        area = data.iloc[minval]["area"]
    elif agg == "max":
        minval = data[mask][col].argmax()
        area = data.iloc[minval]["area"]
    return area
