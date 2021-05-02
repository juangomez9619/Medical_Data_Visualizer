import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
def overweight_calculator(weight, height):
    BMI = weight / ((height/100))**2
    if BMI > 25:
        return 1
    return 0
df['overweight'] = df.apply(lambda row: overweight_calculator(row.weight, row.height),
axis = 1)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
def normalize_cholesterol_gluc(variable):
    if variable == 1:
        return 0
    return 1
df['cholesterol'] = df.apply(lambda row: normalize_cholesterol_gluc(row['cholesterol']), axis=1)
df['gluc'] = df.apply(lambda row: normalize_cholesterol_gluc(row['gluc']), axis=1)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat =  pd.melt(df,id_vars=['cardio'],value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = pd.DataFrame(df_cat.groupby(by = ['cardio','variable','value'])['value'].count())
    df_cat.columns= ['total']
    df_cat.reset_index(inplace=True)
    # Draw the catplot with 'sns.catplot()'
    graph = sns.catplot(x="variable", y="total", hue="value",col="cardio", data=df_cat, kind='bar')
    # Do not modify the next two lines
    fig = graph.fig
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = None

    # Calculate the correlation matrix
    corr = None

    # Generate a mask for the upper triangle
    mask = None



    # Set up the matplotlib figure
    fig, ax = None

    # Draw the heatmap with 'sns.heatmap()'



    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
