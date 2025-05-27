def start_command():
    return 'Привет, я телеграм-бот. Я могу помочь оптимизировать инвестиционный портфель или сделать прогноз на стоимость криптовалютных активов. Что вы хотите сделать?'

def start_make_portfolio():
    return 'Введите криптовалютные пары формата "BTC/USDT", которые будут входить в портфель.\nПосле того как введёте все пары нажмите "Стоп".\nДля того чтобы вернуться назад нажмите "Главное меню".'

def start_make_predict():
    return 'Введите криптовалютные пары формата "BTC/USDT", на которые вы хотите получить прогноз.\nПосле того как введёте все пары нажмите "Стоп".\nДля того чтобы вернуться назад нажмите "Главное меню".'

def return_main():
    return 'Выберете, что хотите сделать.'

def make_portfolio(portfolio):
    ans = ['Ваш портфель:']
    for pair, info in portfolio.items():
        if info['weight']*100 > 0.1:
            ans.append(f"{pair.split('.')[0]}: {info['allocation']:.2f} $ ({info['weight']:.2%})")
    return '\n'.join(ans)

def make_predict(results):
    ans = []
    for pair, info in results.items():
        # Формируем первую строку с количеством дней и парой
        ans.append(f"\n📈 Прогноз на {info['forecast_horizon']} дней для {pair}:")

        forecast_series = info['forecast']
        forecast = '\n'.join(
            f"{date.strftime('%Y-%m-%d')}: {value:.3f}"
            for date, value in zip(forecast_series.index, forecast_series.values)
        )

        # Добавляем прогноз к списку ответов
        ans.append(forecast)

    return '\n'.join(ans)  # Возвращаем готовую строку
