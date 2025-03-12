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