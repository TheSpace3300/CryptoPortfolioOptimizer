def start_command():
    return 'Привет, я телеграм-бот. Я могу помочь оптимизировать инвестиционный портфель или сделать прогноз на стоимость криптовалютных активов.'


def start_make_portfolio():
    return 'Введите криптовалютные пары формата "BTC/USDT", которые будут входить в портфель.\nДля отмены ввода напишите "отмена".'

def predict_active():
    return 'Введите криптовалютные пары формата "BTC/USDT", на которые вы хотите получить прогноз.\nДля отмены ввода напишите "отмена".'


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


def bad_stock():
    return 'Извините, данная акция не найдена'


def bad_responce():
    return 'Ошибка считывания, попробуйте ещё раз'


def make_portfolio(portfolio):
    ans = ['Твой портфель:']
    for ticker, details in portfolio.items():
        if details['weight']*100 > 0.1:
            ans.append(f"{ticker.split('.')[0]}: {details['allocation']:.2f} руб. ({details['weight']:.2%})")
    return '\n'.join(ans)