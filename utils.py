from pathlib import Path

# Import of bank holidays (jours fériés) in France
# Source : https://pypi.org/project/holidays/
# Licence : MIT License (MIT)
import holidays
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from skrub import DatetimeEncoder, GapEncoder, TableVectorizer

# Import of school holidays by zone in france
# Source : https://pypi.org/project/vacances-scolaires-france/
# Licence : MIT License (MIT)
from vacances_scolaires_france import SchoolHolidayDates
from xgboost import XGBRegressor

target_col = "log_bike_count"


###############################################################################################################################
########################################################### UTILS #############################################################
###############################################################################################################################


def covid_period(date):
    confinement_start = pd.Timestamp("2020-10-30")
    confinement_end = pd.Timestamp("2020-12-15")
    couvre_feu_1_start = pd.Timestamp("2020-12-15")
    couvre_feu_1_end = pd.Timestamp("2021-01-15")
    couvre_feu_2_start = pd.Timestamp("2021-01-16")
    couvre_feu_2_end = pd.Timestamp("2021-06-20")
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
    X["month_day"] = X["date"].dt.day
    X["week_day"] = X["date"].dt.day_of_week + 1
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["hour"] = X["date"].dt.hour
    years = X["year"].drop_duplicates().values.tolist()
    french_holidays = set(holidays.country_holidays("FR", years=years))
    X["is_holiday"] = X["date"].dt.date.isin(french_holidays).astype(int)
    X["covid_state"] = X["date"].apply(covid_period)
    ans = X["date"].dt.year.unique()
    holiday_dates = set()
    for an in ans:
        holiday_dates.update(
            date
            for date, info in SchoolHolidayDates()
            .holidays_for_year_and_zone(an, "C")
            .items()
        )
    X["is_school_holiday"] = X["date"].dt.date.isin(holiday_dates).astype(int)

    X = X.drop(columns=["date"])
    return X


def _merge_external_data(X):
    file_path = Path(__file__).parent / "external_data//external_data.csv"

    df_ext = pd.read_csv(file_path, parse_dates=["date"])
    df_ext["date"] = pd.to_datetime(df_ext["date"]).astype("datetime64[us]")

    for col in ["rr24", "rr12", "rr3"]:
        df_ext[col] = np.log1p(df_ext[col])

    df_ext["tend"] = (df_ext["tend"] - df_ext["tend"].mean()) / df_ext["tend"].std()
    df_ext["pres"] = (df_ext["pres"] - df_ext["pres"].mean()) / df_ext["pres"].std()

    X = X.copy()
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"),
        df_ext[["date", "pres", "tend", "rr24", "rr12", "rr3"]].sort_values("date"),
        on="date",
        direction="nearest",
    )
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X


def get_model_data(path="data/train.parquet"):

    data = pd.read_parquet(path)
    data.sort_values(["date", "counter_name"], inplace=True)
    y = data[target_col].values
    X = data.drop(["bike_count", target_col], axis=1)

    return X, y


def train_test_temporal(X, y, delta="30 days"):

    cutoff_date = X["date"].max() - pd.Timedelta(delta)
    train_data = X["date"] <= cutoff_date
    X_train, X_valid = X.loc[train_data], X.loc[~train_data]
    y_train, y_valid = y[train_data], y[~train_data]

    return X_train, X_valid, y_train, y_valid


def get_cv(X, y, random_state=0):
    cv = TimeSeriesSplit(n_splits=8)
    rng = np.random.RandomState(random_state)

    for train_idx, test_idx in cv.split(X):
        # Take a random sampling on test_idx so it's that samples are not consecutives.
        yield train_idx, rng.choice(test_idx, size=len(test_idx) // 3, replace=False)


columns_to_drop = [
    "coordinates",
    "counter_id",
    "site_id",
    "site_name",
    "counter_technical_id",
    "latitude",
    "longitude",
]


def drop_columns(X):
    X = X.drop(columns=columns_to_drop, errors="ignore")
    return X


###############################################################################################################################
################################################## PIPELINE  CONSTRUCTION #####################################################
###############################################################################################################################


#################################################  Pipeline Components   ######################################################

# Date encoding
date_encoder = FunctionTransformer(_encode_date)

# Merging meteorological data
merge = FunctionTransformer(_merge_external_data)

# Dropping unneccessary columns
drop_cols_transformer = FunctionTransformer(drop_columns)

# Preprocessor of the columns
table_vectorizer = TableVectorizer(
    datetime=DatetimeEncoder(resolution="month", add_total_seconds=False),
    high_cardinality=GapEncoder(random_state=42),
    n_jobs=-1,
)

#################################################      Pipeline        ######################################################


def xgb_vectorized_no_date_encoding():

    regressor = XGBRegressor(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=10,
        random_state=42,
        tree_method="hist",
        enable_categorical=True,
    )
    pipe = make_pipeline(
        merge, date_encoder, drop_cols_transformer, table_vectorizer, regressor
    )

    return pipe
