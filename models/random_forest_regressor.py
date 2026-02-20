# models/random_forest_regressor.py
from sklearn.ensemble import RandomForestRegressor

def train_rand_forest_reggr(X_train, y_train):

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model