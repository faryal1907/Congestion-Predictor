# 🛣️ CS-245 Machine Learning Course Project: Micro-Scale Road Congestion Prediction in Lahore

**Group Members**: Faryal Khan (465125), Haleema Imran (481927), Wania Mateen (465219) 
**Theme**: T1: Urban Intelligence & City Futures  
**Problem**: Predict 10-15 min horizon congestion hotspots (0-10 risk) on Lahore roads using heterogeneous data (RTA accidents, Google Mobility, Kaggle Weather, HOTOSM Roads).  
**Stakeholders**: Lahore Traffic Police (hotspot alerts), commuters (reroute), urban planners (simulation).  

## 📋 Quick Start (Reproducible Setup)
Follow these steps to run the full pipeline and PoC locally.

### 1. Pull Repo from GitHub
```
git clone https://github.com/faryal1907/Congestion-Predictor
cd Congestion-Predictor
```

### 2. Environment Setup
- **Conda**:
  ```
  conda env create -f environment.yml
  conda activate ML-PROJECT
  ```


### 3. Create Folders
```
python setup.py
```
- Creates `data/processed/`, `models/`, `poc/`.

### 4. Populate Data/Raw
Download raw datasets and place in `data/raw/`:
- **Quick Zip**: [Google Drive Zip of All Datasets](https://drive.google.com/file/d/1ezZ8MWsj31odnPhtDMB8r_bKQ4fhuGqT/view?usp=sharing) 
- **Individual Links** (if manual):
  - RTA Accidents: [Harvard Dataverse XLSX](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/4VGTDR) (~6.6 MB).
  - Google Mobility (2020-2022): [Export XLSX from Google](https://www.google.com/covid19/mobility/)
  - Weather (2000-2024 + Sep2024-Oct2025): [Kaggle CSVs](https://www.kaggle.com/datasets/rajaisrarkiani/pakistan-historical-weather-data-20002024)
  - OSM Roads: [HDX ZIP](https://data.humdata.org/dataset/hotosm_pak_roads) 


### 5. Run Pipeline Notebooks
Open Jupyter (`jupyter notebook`) > Run notebooks in `src/` sequentially (each self-contained):
1. `data_exploration.ipynb`: Inspect raw (shapes/missing).
2. `data_cleaning.ipynb`: Clean to processed/ CSVs (~40k accidents, 93k mobility, 140k weather, 79k roads).
3. `data_integration.ipynb`: Merge/spatial join, engineer score (~125k rows).
4. `eda.ipynb`: Plots/insights (corr 0.78 incident, midday peaks).
5. `model_pipeline.ipynb`: Train/eval/save PKLs (XGBoost MAE 0.80).
6. `poc_readiness_check.ipynb`: Generate sample_for_poc.csv.

### 6. Launch PoC Dashboard
```
streamlit run poc/poc.py
```
- Opens browser: Tabs for Home (demo), Predict (sliders → score/map), Explore (data/plots), Insights (MAE table).
- Error Handling: Validates inputs, falls back to untuned model if tuned missing.
- Demo (3-5 min): Base/low vs high risk buttons, batch scenarios.

### Troubleshooting
- **No raw data?**: Use Drive zip—extract to data/raw/.
- **Geopandas error?**: `conda install -c conda-forge geopandas`.
- **Model missing?**: Rerun model_pipeline.ipynb.
- **Plots not saving?**: Check plots/ in root; paths relative from src/.
- **Windows paths?**: Backslashes OK in os.path.join.

## 📊 Project Highlights
- **Data**: 40k incidents + 93k mobility + 140k weather + 79k roads → 125k merged (15-min resample, spatial join <0.5km).
- **Engineering**: Score = density * mobility surge * precip multiplier (scaled /2 cap for sensitivity, incident corr 0.78).
- **ML**: XGBoost tuned MAE 0.80 (TimeSeriesSplit, robust to 50% zero scores).
- **Caveats**: Skew/sparsity (median 0 score bias low predictions—realistic for sparse events); expand tuning for production.


## 🛠️ Files Structure
- **data/raw/**: Input datasets (XLSX/CSVs/SHP).
- **data/processed/**: Cleaned CSVs (merged.csv), plots PNGs.
- **models/**: PKLs (XGBoost_Tuned.pkl etc.).
- **poc/poc.py**: Streamlit app.
- **src/**: Notebooks (.ipynb, self-contained).
- **environment.yml**: Conda env.
- **requirements.txt**: Pip deps.
