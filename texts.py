def start_command():
    return 'Привет, я телеграм-бот. Я могу помочь оптимизировать инвестиционный портфель или сделать прогноз на стоимость криптовалютных активов. Что вы хотите сделать?'


def start_make_portfolio():
    return 'Введите криптовалютные пары формата "BTC/USDT", которые будут входить в портфель.\nПосле того как введёте все пары нажмите "Стоп".\nДля того чтобы вернуться назад нажмите "Главное меню".'

def start_make_predict():
    return 'Введите криптовалютные пары формата "BTC/USDT", на которые вы хотите получить прогноз.\nПосле того как введёте все пары нажмите "Стоп".\nДля того чтобы вернуться назад нажмите "Главное меню".'


def request_money():
    return 'Введите сумму, на которую будет составляться портфель.\nСумма должна быть более 1000 USD и не превышать 10 млн USD.\nДля отмены ввода напишите "отмена".'


def return_main():
    return 'Выберете, что хотите сделать.'


def request_stock_name():
    return 'Введите обозначение акции, по которой вы хотите получить информацию\nДля отмены ввода напишите "отмена"'


def low_money():
    return 'Введённая сумма выходит за указаные границы'


def is_not_digit():
    return 'Вы ввели не число, попробуйте ещё раз'


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

        # Формируем строки прогноза
        if hasattr(info['forecast'], 'tolist'):
            forecast = '\n'.join(f"{x:.3f}" for x in info['forecast'].tolist())  # Преобразуем в список строк
        else:
            forecast = f"{info['forecast']:.3f}"  # На случай, если это одно значение

        # Добавляем прогноз к списку ответов
        ans.append(forecast)

    return '\n'.join(ans)  # Возвращаем готовую строку
