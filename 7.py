plt.figure(figsize=(10, 6))
plt.plot(range(len(data)), data, label="Исходные данные")

# Прогноз на тест
plt.plot(range(len(data) - 7, len(data)), test, label="Прогноз (тест)")

# Прогноз в будущее
future_index = range(len(data), len(data) + 7)
plt.plot(future_index, forecast, label="Прогноз (будущее)", linestyle='--')

plt.axvline(len(data) - 7, color='gray', linestyle='--', label='Начало теста')
plt.axvline(len(data), color='black', linestyle='--', label='Будущее')
plt.legend()
plt.title(f"{model_name}: тестовый прогноз и прогноз в будущее")
plt.grid(True)
plt.show()