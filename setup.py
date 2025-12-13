"""
Run via: python setup.py
"""

import os
import warnings
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from xgboost import XGBRegressor
import joblib

import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely import wkt

import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl  # XLSX support

warnings.filterwarnings("ignore")

# Paths (relative to project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_PATH = os.path.join(DATA_DIR, "raw")
PROCESSED_PATH = os.path.join(DATA_DIR, "processed")
MODELS_PATH = os.path.join(BASE_DIR, "models")
PLOTS_PATH = os.path.join(BASE_DIR, "plots")

os.makedirs(RAW_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)


print("Setup complete! Paths ready.")
print(f"Raw: {RAW_PATH}")
print(f"Processed: {PROCESSED_PATH}")
print(f"Models: {MODELS_PATH}")
print(f"Plots: {PLOTS_PATH}")
