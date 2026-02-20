# models/lightGBM.py

from lightgbm import LGBMRegressor

def train_lightgbm(
        X_train,
        y_train
):
    """
     Обучает LightGBM модель (аналог CatBoost, но быстрее).

     Используем базовые параметры:
     - небольшая глубина
     - много деревьев
     - оптимизация MAE
     """

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model