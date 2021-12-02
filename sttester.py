#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 22:54:57 2021

@author: ishansheth
"""

from joblib import load
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import *

st.title('Using Machine Learning to Predict the Outcome of NBA Games')
st.markdown("""
This app uses various machine learning algorithms to predict wins and losses for NBA regular season games.
Choose your season and algorithm to test out the predictions.
You Can also visit the future predictions page to see what the models think about future games.

By Ishan Sheth
""")

log1 = load("logreg_model_3.joblib")
log2 = load("logreg_model_5.joblib")
grad = load("gradboosting_model_3.joblib")
rand = load('randforest_model_5.joblib')
pages = st.sidebar.selectbox('Site Pages', ('Past Predictions', 'Future Predictions', 'Next Steps'))

if pages == 'Past Predictions':
    st.header('Past Predictions from 2018 to Now')
    st.sidebar.header('User Input Features')
    seasons = st.sidebar.selectbox('Select Season', ('2018', '2019', '2020', '2021'))
    
    final2 = pd.read_csv('bigstdata.csv')
    grouped = final2.groupby(final2.SEASON_ID)
    
    features = ['avg_WL', 'avg_NET_RATING', 'avg_PIE', 'opp_avg_NET_RATING', 'opp_avg_PIE', 'MATCHUP']
    feat_list = ['MATCHUP', 'avg_WL', 'avg_FG3M', 'avg_FG3A', 'avg_FG3_PCT', 'avg_BLK', 'avg_OFF_RATING', 'avg_NET_RATING', 'avg_AST_TOV', 'avg_AST_RATIO', 'avg_TM_TOV_PCT', 'avg_PIE', 'opp_avg_WL', 'opp_avg_FG3M', 'opp_avg_DREB', 'opp_avg_REB', 'opp_avg_BLK', 'opp_avg_NET_RATING', 'opp_avg_AST_TOV', 'opp_avg_PIE']
    
    
    if seasons == '2018':
        df = grouped.get_group(22018.0)
    elif seasons == '2019':
        df = grouped.get_group(22019)
        df = df.reset_index()
    elif seasons == '2020':
        df = grouped.get_group(22020.0)
        df = df.reset_index()
    elif seasons == '2021':
        df = pd.read_csv('stdata.csv')
        
    df3 = df[features]
    agree = st.checkbox("Show Feature Table")
    if agree:
        df3
    
    df2 = df[['GAME_DATE', 'WL', 'TEAM_NAME', 'opp_TEAM_NAME']]
    option = st.sidebar.selectbox('Select Model Type', ('Logistic Regression C-4', 'Logistic Regression C-0.5', 'Gradient Boosting Classifier', 'Random Forest Classifier'))
    
    
    print(log2, grad, rand)
    
    if option == 'Logistic Regression C-4':
        
        predicted = pd.DataFrame(log1.predict(df3))
        winner = []
        df4 = pd.concat([df2, predicted], axis=1)
        df6 = df4[['GAME_DATE', 'TEAM_NAME', 'opp_TEAM_NAME', 0, 'WL']]
        for index, row in df6.iterrows():
            if row['WL'] == 1.0:
                winner.append(row['TEAM_NAME'])
            else:
                winner.append(row['opp_TEAM_NAME'])
        df6['Actual Winner'] = winner
        proj_winner = []
        for index, row in df6.iterrows():
            if row[0] == 1.0:
                proj_winner.append(row['TEAM_NAME'])
            else:
                proj_winner.append(row['opp_TEAM_NAME'])   
        df6['Projected Winner'] = proj_winner
        
        df7 = df6[['GAME_DATE', 'TEAM_NAME', 'opp_TEAM_NAME', 'Projected Winner', 'Actual Winner']]
        df7
        
        newlist = []
        newlist2 = []
        newlist5 = []
        
        for index, row in df4.iterrows():
            newlist2.append(index)
            if row['WL'] == row[0]:
                newlist.append(index)
            if row['WL']<row[0]:
                newlist5.append(index)
    
        
        full_season = len(newlist)/len(newlist2)
        st.subheader('Full Season Prediction Accuracy: '+str(full_season))
        st.subheader('There were '+str(len(newlist5))+' upsets this season')
    
        
        n = st.selectbox('Select n-Value', (3, 5, 10, 15, 25, 50, 60, 70, 80, 90, 100))
        df5 = df4.iloc[0:n]
        newlist3 = []
        newlist4 = []
        
        for index, row in df5.iterrows():
            newlist4.append(index)
            if row['WL'] == row[0]:
                newlist3.append(index)
        ten_game = len(newlist3)/len(newlist4)
        st.write('Last n-Games Prediction Accuracy: '+str(ten_game))
    
    elif option == 'Logistic Regression C-0.5':
        
        predicted = pd.DataFrame(log2.predict(df[feat_list]))
        winner = []
        df4 = pd.concat([df2, predicted], axis=1)
        df6 = df4[['GAME_DATE', 'TEAM_NAME', 'opp_TEAM_NAME', 0, 'WL']]
        for index, row in df6.iterrows():
            if row['WL'] == 1.0:
                winner.append(row['TEAM_NAME'])
            else:
                winner.append(row['opp_TEAM_NAME'])
        df6['Actual Winner'] = winner
        proj_winner = []
        for index, row in df6.iterrows():
            if row[0] == 1.0:
                proj_winner.append(row['TEAM_NAME'])
            else:
                proj_winner.append(row['opp_TEAM_NAME'])   
        df6['Projected Winner'] = proj_winner
        
        df7 = df6[['GAME_DATE', 'TEAM_NAME', 'opp_TEAM_NAME', 'Projected Winner', 'Actual Winner']]
        df7
        
        newlist = []
        newlist2 = []
        newlist5 = []
        
        for index, row in df4.iterrows():
            newlist2.append(index)
            if row['WL'] == row[0]:
                newlist.append(index)
            if row['WL']<row[0]:
                newlist5.append(index)
    
        
        full_season = len(newlist)/len(newlist2)
        st.subheader('Full Season Prediction Accuracy: '+str(full_season))
        st.subheader('There were '+str(len(newlist5))+' upsets this season')
    
        
        n = st.selectbox('Select n-Value', (3, 5, 10, 15, 25, 50, 60, 70, 80, 90, 100))
        df5 = df4.iloc[0:n]
        newlist3 = []
        newlist4 = []
        
        for index, row in df5.iterrows():
            newlist4.append(index)
            if row['WL'] == row[0]:
                newlist3.append(index)
        ten_game = len(newlist3)/len(newlist4)
        st.write('Last n-Games Prediction Accuracy: '+str(ten_game)) 
    
    elif option == 'Gradient Boosting Classifier':
        
        predicted = pd.DataFrame(grad.predict(df3))
        winner = []
        df4 = pd.concat([df2, predicted], axis=1)
        df6 = df4[['GAME_DATE', 'TEAM_NAME', 'opp_TEAM_NAME', 0, 'WL']]
        for index, row in df6.iterrows():
            if row['WL'] == 1.0:
                winner.append(row['TEAM_NAME'])
            else:
                winner.append(row['opp_TEAM_NAME'])
        df6['Actual Winner'] = winner
        proj_winner = []
        for index, row in df6.iterrows():
            if row[0] == 1.0:
                proj_winner.append(row['TEAM_NAME'])
            else:
                proj_winner.append(row['opp_TEAM_NAME'])   
        df6['Projected Winner'] = proj_winner
        
        df7 = df6[['GAME_DATE', 'TEAM_NAME', 'opp_TEAM_NAME', 'Projected Winner', 'Actual Winner']]
        df7
        
        newlist = []
        newlist2 = []
        newlist5 = []
        
        for index, row in df4.iterrows():
            newlist2.append(index)
            if row['WL'] == row[0]:
                newlist.append(index)
            if row['WL']<row[0]:
                newlist5.append(index)
    
        
        full_season = len(newlist)/len(newlist2)
        st.subheader('Full Season Prediction Accuracy: '+str(full_season))
        st.subheader('There were '+str(len(newlist5))+' upsets this season')
    
        
        n = st.selectbox('Select n-Value', (3, 5, 10, 15, 25, 50, 60, 70, 80, 90, 100))
        df5 = df4.iloc[0:n]
        newlist3 = []
        newlist4 = []
        
        for index, row in df5.iterrows():
            newlist4.append(index)
            if row['WL'] == row[0]:
                newlist3.append(index)
        ten_game = len(newlist3)/len(newlist4)
        st.write('Last n-Games Prediction Accuracy: '+str(ten_game)) 
    
    
    elif option == 'Random Forest Classifier':
        
        predicted = pd.DataFrame(rand.predict(df[feat_list]))
        winner = []
        df4 = pd.concat([df2, predicted], axis=1)
        df6 = df4[['GAME_DATE', 'TEAM_NAME', 'opp_TEAM_NAME', 0, 'WL']]
        for index, row in df6.iterrows():
            if row['WL'] == 1.0:
                winner.append(row['TEAM_NAME'])
            else:
                winner.append(row['opp_TEAM_NAME'])
        df6['Actual Winner'] = winner
        proj_winner = []
        for index, row in df6.iterrows():
            if row[0] == 1.0:
                proj_winner.append(row['TEAM_NAME'])
            else:
                proj_winner.append(row['opp_TEAM_NAME'])   
        df6['Projected Winner'] = proj_winner
        
        df7 = df6[['GAME_DATE', 'TEAM_NAME', 'opp_TEAM_NAME', 'Projected Winner', 'Actual Winner']]
        df7
        
        newlist = []
        newlist2 = []
        newlist5 = []
        
        for index, row in df4.iterrows():
            newlist2.append(index)
            if row['WL'] == row[0]:
                newlist.append(index)
            if row['WL']<row[0]:
                newlist5.append(index)
    
        
        full_season = len(newlist)/len(newlist2)
        st.subheader('Full Season Prediction Accuracy: '+str(full_season))
        st.subheader('There were '+str(len(newlist5))+' upsets this season')
    
        
        n = st.selectbox('Select n-Value', (3, 5, 10, 15, 25, 50, 60, 70, 80, 90, 100))
        df5 = df4.iloc[0:n]
        newlist3 = []
        newlist4 = []
        
        for index, row in df5.iterrows():
            newlist4.append(index)
            if row['WL'] == row[0]:
                newlist3.append(index)
        ten_game = len(newlist3)/len(newlist4)
        st.write('Last n-Games Prediction Accuracy: '+str(ten_game)) 
elif pages == 'Next Steps':
    st.header('Future Steps for the Project')
    st.markdown("""
        - Use More Data: Models generally perform better with more data so I would love to add in a few more seasons of data
        - Try New Algorithms: I would love to see if Decision Trees or a Feed-Forward Neural Network will create a better model. 
        - Incorporate Betting Odds: I could use betting odds as a potential feature and see how they influence the accuracy of my models.
                """)
    questions = st.checkbox('')
    if questions:
        st.header('''
                 Thank You for Listening!
                 Any Questions?
                 ''')
elif pages == 'Future Predictions':
    st.header('Predict Future Games')
    st.markdown('Choose a team and an opponent and let the model predict the outcome.')
    teams = st.sidebar.selectbox('Select Home Team', (['Bucks', 'Nets', 'Warriors', 'Lakers', '76ers', 'Magic', 'Timberwolves', 'Knicks', 'Kings', 'Jazz', 'Trail Blazers', 'Bulls', 'Hornets', 'Grizzlies', 'Celtics', 'Wizards', 'Thunder', 'Nuggets', 'Cavaliers', 'Suns', 'Raptors', 'Pistons', 'Pelicans', 'Spurs', 'Rockets', 'Pacers', 'Heat', 'Hawks', 'Clippers', 'Mavericks']))
    opp = st.sidebar.selectbox('Select Away Team', (['Bucks', 'Nets', 'Warriors', 'Lakers', '76ers', 'Magic', 'Timberwolves', 'Knicks', 'Kings', 'Jazz', 'Trail Blazers', 'Bulls', 'Hornets', 'Grizzlies', 'Celtics', 'Wizards', 'Thunder', 'Nuggets', 'Cavaliers', 'Suns', 'Raptors', 'Pistons', 'Pelicans', 'Spurs', 'Rockets', 'Pacers', 'Heat', 'Hawks', 'Clippers', 'Mavericks']))
    option = st.sidebar.selectbox('Select Model Type', ('Logistic Regression C-4', 'Logistic Regression C-0.5', 'Gradient Boosting Classifier', 'Random Forest Classifier'))
    if teams == opp:
        st.subheader('Error: You Selected the Same Team Twice')
    else:
        final = pd.read_csv('final.csv')
        grouped = final.groupby(final.TEAM_NAME)
        row1 = grouped.get_group(teams)
        row1 = row1.reset_index()
        row2 = grouped.get_group(opp)
        for cols in row2.columns:
            row2 = row2.rename(columns={cols: 'opp_'+cols})
        row2 = row2.reset_index()
        feats1 = row1[['WL', 'NET_RATING', 'PIE']]
        feats2 = row2[['opp_NET_RATING', 'opp_PIE']]
        feats = pd.concat([feats1, feats2], axis=1)
        feats['MATCHUP'] = 1
        feat_list1 = row1[['WL', 'FG3M', 'FG3A', 'FG3_PCT', 'BLK', 'OFF_RATING', 'NET_RATING', 'AST_TOV', 'AST_RATIO', 'TM_TOV_PCT', 'PIE']]
        feat_list2 = row2[['opp_WL', 'opp_FG3M', 'opp_DREB', 'opp_REB', 'opp_BLK', 'opp_NET_RATING', 'opp_AST_TOV', 'opp_PIE']]
        feat_list = pd.concat([feat_list1, feat_list2], axis=1)
        feat_list.insert(loc=0, column='MATCHUP', value=1)
        if option == 'Logistic Regression C-4':
            predicted = log1.predict(feats)
            if predicted == 0.0:
                st.subheader('The model believes that '+str(opp)+' will win')
            elif predicted == 1.0:
                st.subheader('The model believes that '+str(teams)+' will win')
        elif option == 'Logistic Regression C-0.5':
            predicted = log2.predict(feat_list)
            if predicted == 0.0:
                st.subheader('The model believes that '+str(opp)+' will win')
            elif predicted == 1.0:
                st.subheader('The model believes that '+str(teams)+' will win')
        elif option == 'Gradient Boosting Classifier':
            predicted = grad.predict(feats)
            if predicted == 0.0:
                st.subheader('The model believes that '+str(opp)+' will win')
            elif predicted == 1.0:
                st.subheader('The model believes that '+str(teams)+' will win')
        elif option == 'Random Forest Classifier':
            predicted = rand.predict(feat_list)
            if predicted == 0.0:
                st.subheader('The model believes that '+str(opp)+' will win')
            elif predicted == 1.0:
                st.subheader('The model believes that '+str(teams)+' will win')        
