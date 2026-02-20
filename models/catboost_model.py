# models/catboost_model.py
from catboost import CatBoostRegressor

# ========= Параметры модели CatBoost самые стандартные, пока на них тестирую ====================
def train_catboost(X_train, y_train):

    # model = CatBoostRegressor(
    #     iterations=500,
    #     learning_rate=0.03,
    #     depth=6,
    #     loss_function="MAE",
    #     random_seed=42,
    #     verbose=False
    # )

    model = CatBoostRegressor(
        iterations=800,
        learning_rate=0.05,
        depth=5,

        loss_function="MAE",
        l2_leaf_reg=20,

        subsample=0.7,
        random_strength=2,

        early_stopping_rounds=50,

        random_seed=42,
        verbose=False
    )

    model.fit(X_train, y_train)



    return model