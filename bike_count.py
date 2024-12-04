import pandas as pd
import numpy as np
import holidays
from pathlib import Path
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from flaml import AutoML
from xgboost import XGBRegressor
from prophet import Prophet


target_col = 'log_bike_count'

columns_to_drop = ['bike_count', 'log_bike_count', 'counter_id', 'coordinates',
                   'counter_technical_id', 'counter_installation_date', 'site_id']

date_cols = ['week_day', 'year', 'month', 'hour', 'is_holiday', 'covid_state', 'month_day', 'is_weekend'] 

categorical_cols = ['counter_name'] # J'enleve site_name 

std_cols = ['t']


## Utils ##

def covid_period(date):
    confinement_start = pd.Timestamp('2020-10-30')
    confinement_end = pd.Timestamp('2020-12-15')
    couvre_feu_1_start = pd.Timestamp('2020-12-15')
    couvre_feu_1_end = pd.Timestamp('2021-01-15')
    couvre_feu_2_start = pd.Timestamp('2021-01-16')
    couvre_feu_2_end = pd.Timestamp('2021-06-20')
    if confinement_start <= date <= confinement_end:
        return 1  # lockdown
    elif couvre_feu_1_start <= date <= couvre_feu_1_end:
        return 2  # first curfew
    elif couvre_feu_2_start <= date <= couvre_feu_2_end:
        return 3  # second curfew
    else:
        return 0


def _encode_date(date): 
    date = date.copy()
    date['month_day'] = date['date'].dt.day
    date['week_day'] = date['date'].dt.day_of_week + 1
    date['year'] = date['date'].dt.year
    date['month'] = date['date'].dt.month
    date['hour'] = date['date'].dt.hour
    date['is_weekend'] = (date['week_day'] >= 6).astype(int)
    years = date['year'].drop_duplicates().values.tolist()
    french_holidays = set(holidays.country_holidays('FR', years=years))
    date['is_holiday'] = (date['date']
                        .dt.date
                        .isin(french_holidays)
                        .astype(int))
    date['covid_state'] = date['date'].apply(covid_period)

    return date.drop(columns= 'date')

def _merge_external_data(X):
    file_path = Path(__file__).parent / "external_data//external_data.csv"

    df_ext = pd.read_csv(file_path, parse_dates=["date"])
    df_ext['date'] = pd.to_datetime(df_ext['date']).astype('datetime64[us]')

    X = X.copy()
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"), df_ext[["date", "t"]].sort_values("date"), on="date", direction='nearest',
    )
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X


def get_model_data(path='data/train.parquet'):

    data = pd.read_parquet(path)
    data.sort_values(['date', 'counter_name'], inplace=True)
    y = data[target_col].values
    X = data.drop(columns_to_drop, axis=1)

    return X, y


def train_test_temporal(X, y, delta='30 days'):
    
    cutoff_date = X['date'].max() - pd.Timedelta(delta)
    train_data = (X['date'] <= cutoff_date)
    X_train, X_valid = X.loc[train_data], X.loc[~train_data]
    y_train, y_valid = y[train_data], y[~train_data]

    return X_train, X_valid, y_train, y_valid



###############################################################################################################################
################################################## PIPELINE  CONSTRUCTION #####################################################
###############################################################################################################################

## Pipeline Components ##

scaler = StandardScaler() 

date_encoder = FunctionTransformer(_encode_date)

categorical_encoder = OneHotEncoder(handle_unknown='error')

merge = FunctionTransformer(_merge_external_data)

preprocessor = ColumnTransformer(
[
    ('scaler', scaler, std_cols),
    ('date', OneHotEncoder(handle_unknown='error'), date_cols),
    ('cat', categorical_encoder, categorical_cols)
]
)

## Pipelines ## 

# Linear Regression
def base_pipeline():

    regressor = LinearRegression()
    pipe = make_pipeline(merge, date_encoder, preprocessor, regressor)

    return pipe

# Ridge
def ridge_pipeline():

    regressor = Ridge()
    pipe = make_pipeline(merge, date_encoder, preprocessor, regressor)

    return pipe

# Random Forest
def rf_tuned_pipeline():

    regressor = RandomForestRegressor(max_depth=20, n_jobs=-1)
    pipe = make_pipeline(merge, date_encoder, preprocessor, regressor)

    return pipe

# XGBoost
def xgb_tuned_pipeline():

    regressor = XGBRegressor(
    learning_rate=0.1,
    n_estimators=100,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1,
    reg_alpha=0,
    random_state=42,
    tree_method='hist'
    )

    pipe = make_pipeline(merge, date_encoder, preprocessor, regressor)

    return pipe

# Grid Search XGBoost
param_grid = {
    'xgbregressor__learning_rate': [0.01, 0.1, 0.2],
    'xgbregressor__n_estimators': [100, 200, 300],
    'xgbregressor__max_depth': [4, 6, 8],
    'xgbregressor__min_child_weight': [1, 3, 5],
    'xgbregressor__subsample': [0.6, 0.8, 1.0],
    'xgbregressor__colsample_bytree': [0.6, 0.8, 1.0],
}

def GS_xgb_pipeline():
    
    regressor = XGBRegressor(
        random_state=42,
        tree_method='hist'  # Utilisation du "histogram method" pour accélérer
    )
    pipe = make_pipeline(merge, date_encoder, preprocessor, regressor)
    return pipe

# Définir le GridSearchCV
def grid_search(pipe):
    
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=5,
        verbose=2,
        n_jobs=-1  
    )

    return grid_search

# Preprocessing Pipeline
def preprocess_pipeline():

    pipe = make_pipeline(merge, date_encoder, preprocessor)

    return pipe