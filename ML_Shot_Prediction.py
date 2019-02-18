#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 14:27:05 2019
Kobe Bryant Shot Analysis ML Project
@author: jujharbedi & abhimanyupatel
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('shot_data.csv')

data_columns = ["action_type	", "combined_shot_type", "game_event_id",	"game_id", 
                "lat","loc_x", "loc_y" ,"lon", "minutes_remaining", "period",
                "playoffs", "season", "seconds_remaining", "shot_distance", 
                "shot_made_flag","shot_type", "shot_zone_area", "shot_zone_basic",
                "shot_zone_range", "team_id", "team_name", "game_date",
                "matchup", "opponent", "shot_id"]

keep_columns = ["action_type	", "combined_shot_type",	"game_event_id", "game_id", 
                "lat", "loc_x", "loc_y" ,"lon", "period", "playoffs", "season", 
                "shot_distance", "shot_made_flag", "shot_zone_area", "opponent",
                "shot_id"]

for i in range(len(data_columns)):
    if not data_columns[i] in keep_columns:
        dataset.drop(data_columns[i], inplace = True, axis = 1)

#Removing nan values for shot_flag (data cleaning)
dataset = dataset[pd.notnull(dataset['shot_made_flag'])]

# Creating temp variables for plotting
datapoints = dataset.values
# Removing all shots taken from beyond half-court for plotting           
datapoints = dataset[(dataset['loc_y'] <= 400)]

# Creating Matrix of Features
X = dataset.values
y = dataset.iloc[:, 12].values

#Encode Categorical Variables
categorical_vars = [ "action_type", "combined_shot_type", "shot_zone_area",
                   "opponent", "season"]
for var in dataset:
    if var in categorical_vars:
        dataset = pd.concat([dataset , pd.get_dummies(dataset[var], prefix=var)], 1)
        dataset  = dataset.drop(var, 1)

# Updating Matrix of features
X = dataset.values

# Splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state = 0)

# Build Random Forest regression model to fit to dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100)
regressor.fit(X_train, y_train)

#Predicting a new result
pred = regressor.predict(X_test)

# Plotting made and missed shots 
def shot_map(dataframe):
    for i in range(len(dataframe["shot_made_flag"])):
        if(dataframe["shot_made_flag"].iloc[i] == 0):
            color = 'red'
        else:
            color = 'green'
        plt.figure(3)
        plt.scatter(dataframe["loc_x"].iloc[i], dataframe["loc_y"].iloc[i], color=color)

def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    from matplotlib.patches import Circle, Rectangle, Arc 
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the 
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax

def scatter_plot_by_category(feat):
    import matplotlib.cm as cm
    flag =  datapoints[pd.notnull(datapoints['shot_made_flag'])]
    alpha = 0.1
    gs = flag.groupby(feat)
    cs = cm.rainbow(np.linspace(0, 1, len(gs)))
    for g, c in zip(gs, cs):
        plt.scatter(g[1].loc_x, g[1].loc_y, color=c, alpha=alpha)
        
plt.figure(figsize=(12,11))
scatter_plot_by_category("shot_zone_area")
plt.title("Kobe Shot Zone Map")
draw_court(outer_lines=True)
# Descending values along the axis from left to right
plt.xlim(300,-300)
plt.ylim(-100,400)
        
# Comparing test results with actual results
X_actual = dataset.loc[dataset['shot_id'].isin(X_test[:, 10])]
for i in range(len(X_test[:, 10])):
    for shot in X_actual["shot_id"]:
        if(X_test[i, 10] == shot):
            temp = X_actual.loc[X_actual['shot_id'] == shot]
            if(pred[i] != temp["shot_made_flag"].values):
                c = 'red'
            else:
                c = 'green'
            plt.figure(2)
            plt.scatter(X_test[i, 3], X_test[i, 4], color=c)
            
draw_court(outer_lines=True)
# Descending values along the axis from left to right
plt.xlim(300,-300)
plt.ylim(-100,400)
plt.title("Plot of Predicted Shot Results")            

shot_map(dataset)
plt.title("Made vs Missed Shots") 
draw_court(outer_lines=True)
plt.xlim(300,-300)
plt.ylim(-100,400)        
plt.show()
