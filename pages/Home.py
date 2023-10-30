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

st.set_option('deprecation.showPyplotGlobalUse', False) #Turn off deprecation warnings for heatmap visualisation

st.title('London housing data')

          
st.write("A tool allowing you to compare house prices across london boroughs in combination with a variety of other relevant factors")

##### LOAD DATA ######

@st.cache_data
def load_data(data_url):
    return pd.read_csv(data_url)

yearly_data_url = "C:/Users/Administrator/Documents/GitHub/StreamlitApp/data/housing_in_london_yearly_variables.csv"
monthly_data_url = "C:/Users/Administrator/Documents/GitHub/StreamlitApp/data/housing_in_london_monthly_variables.csv"
  
#st.subheader("Loading Data")
data_load_state = st.text('Loading data...')
data = load_data(yearly_data_url)
data_monthly = load_data(monthly_data_url)
data_load_state.text("Data loaded!")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data_monthly)

##### PROCESS DATA ######
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
    return data
#mean salary and median salary still problematic

data = process_yearly_data(data)
def agg_monthly_data(data):
    #Extract year
    data["date"] = pd.to_datetime(data["date"]).dt.year
    #Aggregate data to prepare for merging with 
    data = data.groupby(["area", "date"]).agg({'average_price': 'mean', 'houses_sold': 'sum', 'no_of_crimes': 'sum', 'code': 'max'})
    return data

data_monthly = agg_monthly_data(data_monthly)


#GEO DATA
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

#scrape borough coords
url = "https://en.wikipedia.org/wiki/List_of_London_boroughs" 
df = get_borough_coords(url)

#Convert latitude and longitude to positive/negative decimals
df['latitude'] = df['latitude'].apply(convert_coords, type = "lat")
df['longitude'] = df['longitude'].apply(convert_coords, type="lon")

#Calculate distance to centre of city using Haversin formula
def dist_centre(latitude, longitude):
    centre = (51.49722222222222,-0.13722222222222222) #Coordinates for Westminster borough
    return geodesic(centre,(latitude, longitude)).km


df["distance_to_centre"] = df.apply(lambda row: dist_centre(row['latitude'], row['longitude']), axis=1)
df["distance_to_centre"] = df["distance_to_centre"].astype(float)
df = df[["Borough", "latitude", "longitude", "distance_to_centre"]]

#Format borough names
df['Borough'] = df['Borough'].str.extract(r'^(.*?)(?:\[|$)')
df['Borough'] = df['Borough'].apply(lambda x: " ".join([a.lower() for a in x.split()]))
df = df.rename(columns={"Borough":"area"})

#join
merged_data = data.merge(data_monthly, on=['area', 'date'], how='left')
merged_data = merged_data.merge(df, on="area", how='left')


if st.checkbox('Show processed data'):
    st.subheader('Processed data')
    st.write(merged_data)

##### DATA EXPLORATION #####

#pr = df.profile_report()
#st_profile_report(pr)

# Define coloums of interest for widgets
col_of_int = ['median_salary', 'mean_salary', 'life_satisfaction',
       'recycling_pct', 'population_size', 'number_of_jobs',    
       'area_size', 'no_of_houses', 'average_price', 'distance_to_centre']

#Widget for user to choose areas to compare
areas = data["area"].values
chosen_areas = st.multiselect("Pick areas to compare", areas)
filtered_data = data[data['area'].isin(chosen_areas)]

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
def create_heatmap(df, columns):
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[columns].corr()
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='viridis', vmin=-1, vmax=1)
    st.pyplot()
#plot = sns.heatmap(merged_data[col_of_int], annot=True, fmt="g", cmap='viridis')
create_heatmap(merged_data, col_of_int)
#
st.write("As can be seen here, a few variables like population size, number of jobs, area size and number of houses are understandably all very highly correlated. Mean salary and median salary also have a high correlation, though not as high - their difference gives important information on the spread of incomes (and hence inequality).")


##### REGRESSION MODEL ######

st.subheader("Modelling house prices with regression")

data_to_model = merged_data
data_to_model = data_to_model[data_to_model["borough_flag"]==1]
data_to_model = data_to_model.dropna()

# Define dependent variable
response_var = "average_price"
Y = data_to_model[response_var]

# Define independent variables
predict_vars = ['life_satisfaction', 'median_salary',
        'recycling_pct', 'population_size', 'no_of_crimes',
        'area_size', 'no_of_houses'
       ]

# Adding a constant to the model (intercept)

X = sm.add_constant(data_to_model[predict_vars])

# Fit the regression model
model = sm.OLS(np.asarray(Y), np.asarray(X)).fit()

# Print the summary of regression
#st.write(model.summary())
summary = model.summary2().tables[0]
summary = summary.iloc[1:]
summary.columns = ["Attribute", "Value", "Attribute ", "Value "]
summary.index = summary.iloc[:,0]
summary = summary.iloc[:, 1:]
st.write(summary)
st.write("The model explains 62.8% of the variance in the dependent variable (housing prices), which suggests a moderate degree of predictive power of the variables. \n\nAll of the predictors seem to be statistically significant with p-values less than 0.05\n\nThe model diagnostics indicate potential issues with the normality of residuals, given the low p-values for the Omnibus and Jarque-Bera tests, and the high Kurtosis value. The condition number is quite high, suggesting potential multicollinearity issues.")

##
coefficients = model.summary2().tables[1]
coefficients.index = ["constant"] + predict_vars
st.write(coefficients)
st.write("Life satisfaction seems to have a disproportional effect on the model, but this can be explained by the minimal variance of the variable. Mean salary, strangely, is modelled as having a negative effect on average house prices . This could be explained by colinnearity with another varibale")


##### MAP #####
st.subheader("Visualising variation on a map")
map_col = st.selectbox("Pick a variable", col_of_int)
year = st.selectbox("Pick a year", merged_data["date"].unique())

#col = "average_price"
#year = 2018
def scale_to_max(df, col, type="size"):
    df = df[df["borough_flag"] == 1]
    if type == "size":
        df["size"] = 3000*df[col]/df[col].max()
    elif type == "colour":
        col_max = df[col].max()
        df["colour"] =df[col].apply(lambda x: int(255*x/col_max))
        df["colour"] =df["colour"].apply(lambda x: (x, abs(x-255),0))
    else:
        print("Choose a valid type!")
    return df
type_grad = "size"
df = scale_to_max(merged_data, map_col, type=type_grad)
df = df[df["date"] == year]
df = df[df["area"] != "city of london"]
#st.write(df)
st.map(df, size=type_grad)

##### SOURCE ######

if st.button('Source'):
     st.write('Data drawn from Justinas Cirtautas\' dataset "Housing in London", in turn drawing from London Datastore \n\nLink: https://www.kaggle.com/datasets/justinas/housing-in-london')
else: 
    ""

    