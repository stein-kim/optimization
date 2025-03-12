# Import necessary packages
from pulp import *
import pandas as pd
import numpy as np

# Define model
model = LpProblem('Minimum Distribution Cost Model', LpMinimize)

# Define types of distribution locations
hub = ['CVG', 'AFW']
focus = ['Leipzig', 'Hyderabad', 'San Bernardino']
center = ['Paris', 'Cologne', 'Hanover', 'Bangalore', 'Coimbatore', 'Delhi', 
          'Mumbai', 'Cagliari', 'Catania', 'Milan', 'Rome', 'Katowice', 
          'Barcelona', 'Madrid', 'Castle Donington', 'London', 'Mobile', 
          'Anchorage', 'Fairbanks', 'Phoenix', 'Los Angeles', 'Ontario', 
          'Riverside', 'Sacramento', 'San Francisco', 'Stockton', 'Denver', 
          'Hartford', 'Miami', 'Lakeland', 'Tampa', 'Atlanta', 'Honolulu', 
          'Kahului/Maui', 'Kona', 'Chicago', 'Rockford', 'Fort Wayne', 
          'South Bend', 'Des Moines', 'Wichita', 'New Orleans', 'Baltimore', 
          'Minneapolis', 'Kansas City', 'St. Louis', 'Omaha', 'Manchester', 
          'Albuquerque', 'New York', 'Charlotte', 'Toledo', 'Wilmington', 
          'Portland', 'Allentown', 'Pittsburgh', 'San Juan', 'Nashville', 
          'Austin', 'Dallas', 'Houston', 'San Antonio', 'Richmond', 
          'Seattle/Tacoma', 'Spokane']

# Import capacity, cost, and demand information and create data frames
# Cost for shipping from hub or focus city to distribution centers
df = pd.read_csv('opt_df_cost_centers.csv')
df = df.replace(np.nan, '0')
df_hub_center = df[['Centers', 'CVG', 'AFW']]
# Change appropriate columns to numeric
df_hub_center[['CVG', 'AFW']] = df_hub_center[['CVG', 'AFW']].apply(pd.to_numeric, errors='coerce') 
print(df_hub_center)
df_focus_center = df[['Centers', 'Leipzig', 'Hyderabad', 'San Bernardino']]
# Change columns to numeric
df_focus_center[['Leipzig', 'Hyderabad', 'San Bernardino']] = df_focus_center[['Leipzig', 'Hyderabad', 'San Bernardino']].apply(pd.to_numeric, errors='coerce')
print(df_focus_center)

# Cost for shipping from hub to focus city
df = pd.read_csv('opt_df_cost_focus.csv')
df = df.replace(np.nan, '0')
df_hub_focus = df[['city', 'CVG', 'AFW']]
df_hub_focus[['CVG', 'AFW']] = df_hub_focus[['CVG', 'AFW']].apply(pd.to_numeric, errors='coerce')
print(df_hub_focus)

# Hub and focus city capacities
df = pd.read_csv('opt_df_capacity.csv')
df = df.replace(np.nan, '0')
df['Capacity'] = df['Capacity'].str.replace(',', '', regex=False).astype(int)
df_hub_cap = df.loc[df['City'].isin(['CVG', 'AFW'])]
print(df_hub_cap)
df_focus_cap = df.loc[df['City'].isin(['Leipzig', 'Hyderabad', 'San Bernardino'])]
df_focus_cap = df_focus_cap.reset_index(drop=True)
print(df_focus_cap)

# Demand
df_demand = pd.read_csv('opt_df_demand.csv')
df_demand['Demand'] = df_demand['Demand'].str.replace(',', '', regex=False).astype(int)
print(df_demand)

# Create variable dictionaries
# Hub to focus city (xij)
xij = LpVariable.dicts('hub_to_focus_', [(i,j) for i in hub for j in focus], lowBound=0, upBound=None, cat='Integer')

# Hub to center (yik)
yik = LpVariable.dicts('hub_to_center_', [(i,k) for i in hub for k in center], lowBound=0, upBound=None, cat='Integer')

# Focus to center (zjk)
zjk = LpVariable.dicts('focus_to_center_', [(j,k) for j in focus for k in center], lowBound=0, upBound=None, cat='Integer')

# Define the model
model += (lpSum([df_hub_focus.loc[df_hub_focus['city'] == j, i].values[0] * 
                 xij[(i, j)] for i in hub for j in focus if (df_hub_focus['city'] == j).any()]) +
          lpSum([df_hub_center.loc[df_hub_center['Centers'] == k, i].values[0] * 
                 yik[(i, k)] for i in hub for k in center if (df_hub_center['Centers'] == k).any()]) +
          lpSum([df_focus_center.loc[df_focus_center['Centers'] == k, j].values[0] * 
                 zjk[(j, k)] for j in focus for k in center if (df_focus_center['Centers'] == k).any()]))

# Define the constraints:
# Hub capacities
for i in hub:
    model += lpSum([xij[(i, j)] for j in focus]) + lpSum([yik[(i, k)] for k in center]) <= df_hub_cap.loc[df_hub_cap['City'] == i, 'Capacity'].values[0]

# Quantity into focus cities
for j in focus:
  model += lpSum([xij[(i, j)] for i in hub]) <= df_focus_cap.loc[df_focus_cap['City'] == j, 'Capacity'].values[0]

#Quantity out of focus cities
for j in focus:
  model += lpSum([zjk[(j, k)] for k in center]) == lpSum([xij[(i, j)] for i in hub])

# Center demand
for k in center:
  model += lpSum([yik[(i, k)] for i in hub]) + lpSum([zjk[(j, k)] for j in focus]) == df_demand.loc[df_demand['City'] == k, 'Demand'].values[0]

# Solve the model
model.solve()

# Check whether optimal solution was found
print(LpStatus[model.status])

# Check variable assignments
total_var = 0
for v in model.variables():
  print(v.name, "=", v.varValue)
  total_var += 1
print("Total value:", total_var)

# Check that constraints were included
for name, constraint in model.constraints.items():
    print(f"{name}: {constraint}")

# Check result
print('Minimum distribution cost = $', value(model.objective))