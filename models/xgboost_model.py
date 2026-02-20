# models/xgboost_model.py
from xgboost import XGBRegressor

def train_xgboost(
        X_train,
        y_train
):
    """
        Обучение модели XGBoostRegressor для прогноза временных рядов.

        Используется как альтернативный ML baseline рядом с CatBoost и RF.

        Параметры подобраны как "безопасные стартовые":
            - не слишком глубокие деревья
            - небольшой learning_rate
            - достаточно много деревьев
            - регуляризация против переобучения

        Возвращает обученную модель.
    """

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,

        subsample=0.8,
        colsample_bytree=0.8,

        objective="reg:absoluteerror",
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


