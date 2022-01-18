from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px 
import matplotlib.pyplot as plt

#import preprocessing packages
from sklearn.model_selection import train_test_split

#import models from SciKit Learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

#import Error Metrics from Scikit Learn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

@st.cache
def load_dataset(filename):
    loaded_dataset = pd.read_csv(filename)
    return loaded_dataset
    

#set page icon, title, and layout
st.set_page_config(page_title = "D-FLY",
                   page_icon = ":airplane:",
                   layout = "wide",
) 


#read & display dataset into streamlit
delay_data = load_dataset("C:/Users/SHI ERN/Documents/UM/sem5/project/2018_flight_delays.csv")
model_delay_data = pd.read_csv("C:/Users/SHI ERN/Documents/UM/sem5/project/model_training_result.csv")
date_location_carrier = pd.read_csv("C:/Users/SHI ERN/Documents/UM/sem5/project/flight_date_carrier_location.csv")
    
#create a side bar to choose home page, dataset page or evaluation page
pageselection = st.sidebar.selectbox("Please choose action:", ["Home", "Dataset", "Evaluation"])

#home page
if pageselection == "Home":
    
    #brief introduction
    st.title("Welcome to D-FLY! :airplane:")
    st.text("D-FLY is a Data Science project that aims to provide users to access to domestic flight related data in US anytime.")
    st.markdown("**What does *D-FLY* stands for** :question:")
    st.markdown("**D**: delay or domestic")
    st.markdown("**FLY** : motion of flight moving in the air")
    st.header(" ")
    
    #about author & other related links used in this project
    col1, col2, col3 = st.columns([2,4,5]) 
    with col1:
        st.markdown("**About Author**")
        image = Image.open('C:/Users/SHI ERN/Pictures/cse.jpg')
        st.image(image)
        
    with col2:
        st.header(" ")
        st.subheader(" ")
        st.write("CHAN SHI ERN")
        st.write("Bachelor of Computer Science(Data Science)")
        st.write("University Malaya")
        st.subheader(" ")
        st.write("For any enquiries, feel free to contact me via")
        st.markdown(":mailbox: : *shiern304@gmail.com*")
        
    with col3: 
        st.markdown("**Related Links**")
        st.write("Github link: ")
        st.write("Data source: ")
        st.write("https://www.kaggle.com/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018")

   
#dataset page
if pageselection == "Dataset":
    
    #display details of dataset (eg: total number of flights, average delays)
    topleft_col, topright_col = st.columns([2,1])
    
    st.title("Data Exploration :mag_right:")

    with topleft_col:
        st.subheader("Total Number of Flights:")
        total_flights = len(delay_data["ARR_DELAY"].index) #calculate total number of rows in dataframe
        st.subheader(f"{total_flights:,}")
        
    with topright_col:
        st.subheader("Average Delay:")
        st.text("(In the scale of 1-10)")
        ave_d = delay_data["ARR_DELAY"].mean()
        ave_delays = round(ave_d, 1) *10 #round up to 1 decimal then get the integer value 
        ave_level = ":airplane:" * int(ave_delays)
        st.subheader(f"{ave_level}")
    
    #display charts
    st.subheader("Which type of chart you wish to explore?")
    chart_rad = st.radio("Please choose: ",["Factors related", "Time related"])
    
    if chart_rad == "Factors related":
        group_factors = delay_data[["CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]]
        f_col1, f_col2 = st.columns([3,1])
        with f_col1:
            st.subheader("Bar Chart of Different Delays Factors")
            factor_bar = st.bar_chart(group_factors.sum())
            st.markdown("**The most impacting factor: :boom: NAS_DELAY :boom:**")
            
        with f_col2:
            st.header(" ")
            st.header(" ")
            st.write(group_factors.sum())
        
        f_col3, f_col4 = st.columns([4,1])
        with f_col3:
            st.subheader("Air Line Company")
            st.bar_chart(delay_data["OP_CARRIER"].value_counts())  
            st.markdown("**The most impacting airline carrier: :boom: WN :boom:**")
            
        with f_col4:
            st.header(" ")
            st.header(" ")
            st.write(delay_data["OP_CARRIER"].value_counts())
                  
    if chart_rad == "Time related":
        t_col1, t_col2 = st.columns([3,1])
        with t_col1:
            st.subheader("Which day has the most delay?")
            dayofweek = date_location_carrier["day_of_week_name"].value_counts()
            st.line_chart(dayofweek)
            st.markdown("**The most impacting day: :date: MONDAY :date:**")
        
        with t_col2:
            st.header(" ")
            st.header(" ")
            st.write(date_location_carrier["day_of_week_name"].value_counts())
        
        t_col3, t_col4 = st.columns([4,1])
        with t_col3:
            st.subheader("Which month has the most delay?")
            st.line_chart(date_location_carrier["month"].value_counts())
            st.markdown("**The most impacting month: :date: JULY :date:**")
        
        with t_col4:
            st.header(" ")
            st.header(" ")
            st.write(date_location_carrier["month"].value_counts())

#Evaluation page    
if pageselection == "Evaluation": 
    
    st.title("Evaluation Metrics Result :bar_chart:")
    st.write("Algorithms used:")
    st.write(":pushpin: Logistic Regression")
    st.write(":pushpin: K-Nearest Neighbours")
    st.write(":pushpin: Gaussian Naive Bayes")
    st.write(":pushpin: Support Vector Machine")
    
    #create radio button to allow user to choose the types of evaluation metrics they wish to see
    algo_rad = st.sidebar.radio("Please choose an evaluation metrics", ["Accuracy", "Precision", "Recall", "F1-score"])
    
    #create 2 columns, left side to display the data, right side to display charts
    algo_col, visual_col = st.columns(2) 
    model_delay_data.set_index('Algo', inplace = True)
           
    
    with algo_col:
        if algo_rad == "Accuracy":
            st.header("- A C C U R A C Y -")
            st.write(model_delay_data["Accuracy"])
            st.subheader("Best Accuracy: Support Vector Machine :+1:")
            
        if algo_rad == "Precision":
            st.subheader("- P R E C I S I O N -")
            st.write(model_delay_data["Precision"])
            st.subheader("Best Precision: Support Vector Machine :+1:")
            
        if algo_rad == "Recall":
            st.subheader("- R E C A L L -")
            st.write(model_delay_data["Recall"])
            st.subheader("Best Recall: Support Vector Machine :+1:")
            
        if algo_rad == "F1-score":
            st.subheader("- F 1 - S C O R E -")
            st.write(model_delay_data["F1 Score"])
            st.subheader("Best F1-score: Support Vector Machine :+1:")
    
    
    with visual_col:
        if algo_rad == "Accuracy":
            st.bar_chart(model_delay_data["Accuracy"]) 
            
        if algo_rad == "Precision":
            st.bar_chart(model_delay_data["Precision"]) 
            
        if algo_rad == "Recall":
            st.bar_chart(model_delay_data["Recall"]) 
            
        if algo_rad == "F1-score":
            st.bar_chart(model_delay_data["F1 Score"])
            
    
    
    
        