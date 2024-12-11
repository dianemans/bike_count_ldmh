import pandas as pd
import numpy as np
import holidays
from pathlib import Path
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from skrub import TableVectorizer, DatetimeEncoder
from vacances_scolaires_france import SchoolHolidayDates


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

    X = X.drop(columns=['date'])
    return X

def _merge_external_data(X):
    file_path = Path(__file__).parent / "external_data//external_data.csv"

    df_ext = pd.read_csv(file_path, parse_dates=["date"])
    df_ext['date'] = pd.to_datetime(df_ext['date']).astype('datetime64[us]')

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


###############################################################################################################################
################################################## PIPELINE  CONSTRUCTION #####################################################
###############################################################################################################################


#################################################  Pipeline Components   ######################################################

# Date Encoding 
date_encoder = FunctionTransformer(_encode_date)

# Merging Meteorological Data
merge = FunctionTransformer(_merge_external_data)

drop_cols_transformer = FunctionTransformer(drop_columns)

table_vectorizer = TableVectorizer(
        #specific_transformers=[(drop_cols_transformer, columns_to_drop)], 
        datetime=DatetimeEncoder(resolution='month', add_total_seconds=False),
        n_jobs=-1
    )

#################################################      Pipelines        ######################################################

def xgb_vectorized_no_date_encoding(): 

    #regressor = XGBRegressor(max_depth= 10, min_child_weight= 7, subsample= 0.8972852751497171, colsample_bytree= 0.7366839097750602, reg_alpha= 0.002644395912568715, reg_lambda= 0.00025636265208962237)
    regressor = XGBRegressor(learning_rate=0.1, n_estimators=100, max_depth=10, random_state=42, tree_method='hist', enable_categorical=True)
    
    pipe = make_pipeline(merge, date_encoder, drop_cols_transformer, table_vectorizer, regressor) 

    return pipe

def rf_vectorized_no_date_encoding():

    regressor = RandomForestRegressor(max_depth=10, random_state=42)
    
    pipe = make_pipeline(merge, date_encoder, drop_cols_transformer, table_vectorizer, regressor) 

    return pipe

