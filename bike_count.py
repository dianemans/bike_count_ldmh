import pandas as pd
import holidays
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

target_col = 'log_bike_count'
columns_to_drop = ['bike_count', 'log_bike_count', 'counter_id',
                        'coordinates', 'counter_technical_id',
                          'counter_installation_date']



def _get_season(month):
    if month in [12, 1, 2]:
        return 'winter'
    if month in [3, 4, 5]:
        return 'spring'
    if month in [6, 7, 8]:
        return 'summer'
    else:
        return 'fall'


def _encode_date(date):
    date = date.copy()
    date['month_day'] = date['date'].dt.day
    date['week_day'] = date['date'].dt.day_of_week + 1
    date['year'] = date['date'].dt.year
    date['month'] = date['date'].dt.month
    date['hour'] = date['date'].dt.hour
    date['is_weekend'] = (date['week_day'] >= 6).astype(int)
    date['season'] = date['month'].apply(_get_season)
    years = date['year'].drop_duplicates().values.tolist()
    french_holidays = set(holidays.country_holidays('FR', years=years))
    date['is_holiday'] = (date['date']
                        .dt.date
                        .isin(french_holidays)
                        .astype(int)
    )

    return date.drop(columns= 'date')


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


def base_pipeline():
    date_encoder = FunctionTransformer(_encode_date)
    date_cols = ['month_day', 'week_day', 'year', 'month', 'hour', 'is_weekend', 'season', 'is_holiday']

    categorical_encoder = OneHotEncoder(handle_unknown='infrequent_if_exist')
    categorical_cols = ['counter_name', 'site_name']

    preprocessor = ColumnTransformer(
    [
        ('date', OneHotEncoder(handle_unknown='infrequent_if_exist'), date_cols),
        ('cat', categorical_encoder, categorical_cols)
    ]
    )

    regressor = LinearRegression()

    pipe = make_pipeline(date_encoder, preprocessor, regressor)

    return pipe

def rf_tuned_pipeline():
    date_encoder = FunctionTransformer(_encode_date)
    date_cols = ['month_day', 'week_day', 'year', 'month', 'hour', 'is_weekend', 'season', 'is_holiday']

    categorical_encoder = OneHotEncoder(handle_unknown='infrequent_if_exist')
    categorical_cols = ['counter_name', 'site_name']

    preprocessor = ColumnTransformer(
    [
        ('date', OneHotEncoder(handle_unknown='infrequent_if_exist'), date_cols),
        ('cat', categorical_encoder, categorical_cols)
    ]
    )

    regressor = RandomForestRegressor(max_depth=20, n_jobs=-1)

    pipe = make_pipeline(date_encoder, preprocessor, regressor)

    return pipe


def xgb_tuned_pipeline():
    date_encoder = FunctionTransformer(_encode_date)
    date_cols = ['month_day', 'week_day', 'year', 'month', 'hour', 'is_weekend', 'season', 'is_holiday']

    categorical_encoder = OneHotEncoder(handle_unknown='infrequent_if_exist')
    categorical_cols = ['counter_name', 'site_name']

    preprocessor = ColumnTransformer(
    [
        ('date', OneHotEncoder(handle_unknown='infrequent_if_exist'), date_cols),
        ('cat', categorical_encoder, categorical_cols)
    ]
    )

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

    pipe = make_pipeline(date_encoder, preprocessor, regressor)

    return pipe