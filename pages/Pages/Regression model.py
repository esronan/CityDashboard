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
sys.path.append('C:/Users/Administrator/Documents/GitHub/StreamlitApp/data')
sys.path.append('C:/Users/Administrator/Documents/GitHub/StreamlitApp/utils')
from keys import db_instance, db_user, db_pass, db_name
import utils

st.title("Regression model")

#initiate connector
connector = Connector()

#initiate connection engine
engine = utils.init_connection_engine(db_instance=db_instance, db_user=db_user, db_pass=db_pass, db_name=db_name, connector_instance=connector)

#query from database (first checking if data has been processed)
try:
    hu
    query = "SELECT * FROM processed_data"
    df = pd.read_sql(query, con=engine)
except:
    try: 
        df= pd.read_csv("C:/Users/Administrator/Documents/GitHub/StreamlitApp/data/processed_data.csv")
    except:
        st.write("Please let home tab load first.")

st.subheader("Modelling house prices with regression")
st.write("Before modelling with regression, we applied a few steps of preprocessing. First, all columns with non-numerical data was deleted. Then all rows with missing data were deleted (data integrity was not affected - data was missing only when collection procedures had not been set up - e.g. life satisfaction was first collected in 2011).")
data_to_model = df
data_to_model = data_to_model[data_to_model["borough_flag"]==1]
data_to_model = data_to_model.dropna()


# Define dependent variable
response_var = "average_price"
Y = data_to_model[response_var]

# Define independent variables
predict_vars = ['median_salary',
        'population_density', 'no_of_crimes',
        'distance_to_centre'
       ] 
# 'life_satisfaction', 'recycling_pct', 
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
st.write("The model explains 50.4% of the variance in the dependent variable (housing prices), which suggests a moderate degree of predictive power of the variables. \n\nThe F ratio is far above 1, meaning that there is much more signal to noise in the model, and this is validated with a p-value below 0.05.")

##
coefficients = model.summary2().tables[1]
coefficients.index = ["constant"] + predict_vars
st.write(coefficients)
st.write("Life satisfaction seems to have a disproportional effect on the model, but this can be explained by the minimal variance of the variable. Mean salary, strangely, is modelled as having a negative effect on average house prices. This could be explained by collinnearity with another variable")
