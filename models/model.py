from models.model_selector import generate_training_data, train_selector_model
from services.fetch_data import fetch_ohlcv

# Генерация обучающих данных
pairs = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "DOGE/USDT"]
X_train, y_train = generate_training_data(pairs, fetch_ohlcv)

# Обучение модели выбора
clf = train_selector_model(X_train, y_train)

# Сохраняем модель
print("Модель выбора обучена и сохранена.")