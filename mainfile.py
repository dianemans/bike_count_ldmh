import pandas as pd
import utils as u

### Make Predictions Here ###
X, y = u.get_model_data()
pipe = u.xgb_vectorized_no_date_encoding()
pipe.fit(X, y)


test_data = pd.read_parquet("data/final_test.parquet")
test_pred = pipe.predict(test_data)

test_df = pd.DataFrame({"Id": range(0, len(test_pred)), "log_bike_count": test_pred})
test_df.to_csv("submission.csv", index=False)