import pandas as pd
import numpy as np
import holidays
from pathlib import Path
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from flaml import AutoML
from xgboost import XGBRegressor
from skrub import TableVectorizer, DatetimeEncoder
import optuna 
from vacances_scolaires_france import SchoolHolidayDates
from sklearn.metrics import root_mean_squared_error


'''RESUMÉ DE CE QUI MARCHE LE MIEUX

best pipeline: 
- Merge avec le dataset externe de météo, en incluant les tendances de pression atmosphérique, la pression, 
et les données de pluie sur les dernières 24, 12, et 3 heures. 
- Encoder les dates et utiliser le table vectorizer pour automatiquement preprocess les données de la bonne manière
- XGBoost (non tuné encore)

A PRENDRE EN COMPTE: 
- Potentiellement log-trasnform les données de pluie (en trouvant un moyen d'incorporer tout de meme les données = 0)
- Drop les coordonnées n'a diminué la perf du modèle que de 0.0001, donc autant dire qu'on peut considérer que c'est mieux sans.
- Ajouter site_name: 
'''


target_col = 'log_bike_count'


columns_to_drop = ['coordinates', 'counter_id', 'site_id', 'site_name','counter_technical_id', 'latitude', 'longitude'] #, site_name]
# JESSAYE DE DROP LATI ET LONGI
date_cols = ['week_day', 'year', 'month', 'hour', 'is_holiday', 'covid_state', 'month_day', 'is_school_holiday'] 

categorical_cols = ['counter_name'] #, 'counter_technical_id', 'site_name'] 

std_cols = ['t']


###############################################################################################################################
########################################################### UTILS #############################################################
###############################################################################################################################


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

school_holidays = SchoolHolidayDates()


def _encode_date(X): 
    X = X.copy()
    X['month_day'] = X['date'].dt.day
    X['week_day'] = X['date'].dt.day_of_week + 1
    X['year'] = X['date'].dt.year
    X['month'] = X['date'].dt.month
    X['hour'] = X['date'].dt.hour
    #X['is_weekend'] = (X['week_day'] >= 6).astype(int) 
    years = X['year'].drop_duplicates().values.tolist()
    french_holidays = set(holidays.country_holidays('FR', years=years))
    X['is_holiday'] = (X['date']
                        .dt.date
                        .isin(french_holidays)
                        .astype(int))
    X['covid_state'] = X['date'].apply(covid_period)
    years = X['date'].dt.year.unique()
    holiday_dates = set()
    for year in years:
        holiday_dates.update(
            date for date, info in SchoolHolidayDates().holidays_for_year_and_zone(year, 'C').items()
        )
    X['is_school_holiday'] = X['date'].dt.date.isin(holiday_dates).astype(int)
    #X['is_school_holiday'] = X['date'].apply(is_paris_school_holiday)

    X = X.drop(columns=['date'])
    return X

def _merge_external_data(X):
    file_path = Path(__file__).parent / "external_data//external_data.csv"

    df_ext = pd.read_csv(file_path, parse_dates=["date"])
    df_ext['date'] = pd.to_datetime(df_ext['date']).astype('datetime64[us]')
    
    for col in ['rr24', 'rr12', 'rr3']:
        df_ext[col] = np.log1p(df_ext[col])  # log(x + 1)

    df_ext['tend'] = (df_ext['tend'] - df_ext['tend'].mean()) / df_ext['tend'].std()
    df_ext['pres'] = (df_ext['pres'] - df_ext['pres'].mean()) / df_ext['pres'].std()
    X = X.copy()
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"), df_ext[["date", 'pres', 'tend', 'rr24', 'rr12', 'rr3']].sort_values("date"), on="date", direction='nearest',
    )
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X ## better with 'pres' + 'tend' + 'rr24' + 'rr12' + 'rr3'


def get_model_data(path='data/train.parquet'):

    data = pd.read_parquet(path)
    data.sort_values(['date', 'counter_name'], inplace=True)
    y = data[target_col].values
    X = data.drop(['bike_count', target_col], axis=1)

    return X, y


def train_test_temporal(X, y, delta='30 days'):
    
    cutoff_date = X['date'].max() - pd.Timedelta(delta)
    train_data = (X['date'] <= cutoff_date)
    X_train, X_valid = X.loc[train_data], X.loc[~train_data]
    y_train, y_valid = y[train_data], y[~train_data]

    return X_train, X_valid, y_train, y_valid

def drop_columns(X):
    X = X.drop(columns=columns_to_drop, errors='ignore')
    return X

def fill_na(X):
    X = X.fillna(0)
    return X

###############################################################################################################################
################################################## PIPELINE  CONSTRUCTION #####################################################
###############################################################################################################################


#################################################  Pipeline Components   ######################################################

# Scaler
scaler = StandardScaler() 

# Date Encoding 
date_encoder = FunctionTransformer(_encode_date)

# Categorical Encoding 
categorical_encoder = OneHotEncoder(handle_unknown='error')

# Merging Meteorological Data
merge = FunctionTransformer(_merge_external_data)

drop_cols_transformer = FunctionTransformer(drop_columns)

table_vectorizer = TableVectorizer(
        specific_transformers=[(drop_cols_transformer, columns_to_drop)], 
        datetime=DatetimeEncoder(resolution='month', add_total_seconds=False),
        cardinality_threshold=100,
        n_jobs=-1
    )

# Preprocessing steps
preprocessor = ColumnTransformer(
[
    ('drop_cols', 'drop', columns_to_drop),
    ('scaler', scaler, std_cols),
    ('date', OneHotEncoder(handle_unknown='error'), date_cols),
    ('cat', categorical_encoder, categorical_cols),
], 
remainder='passthrough'
)


fillna_transformer = FunctionTransformer(fill_na)

#################################################      Pipelines        ######################################################


# XGBoost
def xgb_tuned_pipeline():
    regressor = XGBRegressor(learning_rate=0.1, n_estimators=100, max_depth=10, min_child_weight=1,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1, reg_alpha=0,
                             random_state=42, tree_method='hist')
    
    pipe = make_pipeline(merge, date_encoder, preprocessor, regressor)
    return pipe


#Pipeline sans external data

def xgb_tuned_pipeline_no_merge():

    preprocessor1 = ColumnTransformer(
    [
        ('drop_cols', 'drop', columns_to_drop),
        ('date', OneHotEncoder(handle_unknown='error'), date_cols),
        ('cat', categorical_encoder, categorical_cols),
    ]
    )
    
    regressor = XGBRegressor( # Modele réduit
    learning_rate=0.1,
    n_estimators=100,
    max_depth=10,
    random_state=42,
    tree_method='hist'
    )
    pipe = make_pipeline(date_encoder, preprocessor1, regressor)
    return pipe


def xgb_no_encoding():
    preprocessor2 = ColumnTransformer(
    [
        ('drop_cols', 'drop', columns_to_drop),
        ('cat', categorical_encoder, categorical_cols)
    ],
    remainder='passthrough'
    )
    
    regressor = XGBRegressor( # Modele réduit
    learning_rate=0.1,
    n_estimators=100,
    max_depth=10,
    random_state=42,
    tree_method='hist', 
    enable_categorical=True
    )
    pipe = make_pipeline(date_encoder, preprocessor2, regressor)
    return pipe


def xgb_vectorized_no_date_encoding(): # best pipeline yet

    table_vectorizer = TableVectorizer(
        specific_transformers=[(drop_cols_transformer, columns_to_drop)], 
        datetime=DatetimeEncoder(resolution='month', add_total_seconds=False),
        cardinality_threshold=100,
        n_jobs=-1
    )

    regressor = XGBRegressor(max_depth= 10, min_child_weight= 7, subsample= 0.8972852751497171, colsample_bytree= 0.7366839097750602, reg_alpha= 0.002644395912568715, reg_lambda= 0.00025636265208962237)

    pipe = make_pipeline(merge, date_encoder, table_vectorizer, regressor) # ADDED MERGE

    return pipe


def xgb_vectorized_for_optuna(trial=None):

    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3) if trial else 0.1
    max_depth = trial.suggest_int("max_depth", 3, 15) if trial else 10
    n_estimators = trial.suggest_int("n_estimators", 50, 200) if trial else 100

    regressor = XGBRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        tree_method='hist',
        enable_categorical=True,
    )
    '''
    table_vectorizer = TableVectorizer(
        specific_transformers=[(drop_cols_transformer, columns_to_drop)], 
        datetime=DatetimeEncoder(resolution='month', add_total_seconds=False),
        n_jobs=-1
    )
    '''
    pipe = make_pipeline(regressor)
    
    return pipe


def extra_trees():

    table_vectorizer = TableVectorizer(
        specific_transformers=[(drop_cols_transformer, columns_to_drop)], 
        datetime=DatetimeEncoder(resolution='month', add_total_seconds=False),
        n_jobs=-1
    )

    regressor = ExtraTreesRegressor(n_estimators=100, random_state=42, max_depth=10)

    pipe = make_pipeline(merge, date_encoder, fillna_transformer, table_vectorizer, regressor) # ADDED MERGE

    return pipe

###############################################################################################################################
##################################################### Optimisation #############################################################
###############################################################################################################################


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
        tree_method='hist'  
    )
    pipe = make_pipeline(merge, date_encoder, preprocessor, regressor)
    return pipe

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

# Optuna XGBoost

def objective(trial):
    param = {
        'objective': 'reg:squarederror',  # ou 'binary:logistic' pour un problème de classification binaire
        'eval_metric': 'rmse',  # Pour un problème de régression, ou 'logloss' pour classification
        'max_depth': trial.suggest_int('max_depth', 8, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-4, 1e-1),  # L2 regularization term
        'lambda': trial.suggest_loguniform('lambda', 1e-4, 1e-1)  # L1 regularization term
    }
    
    model = xgb.XGBRegressor(**param)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)

    y_pred = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, y_pred)  # ou accuracy_score pour classification

    return rmse

