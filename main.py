import asyncio
import sys
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from aiogram import Bot, Dispatcher, types, Router, F
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.filters import Command, CommandStart
import ccxt

exchange = ccxt.bybit()


from data.download_data import data_forecast
from aiogram.filters import StateFilter
from utils.opt import create_investment_portfolio
from data.config import TOKEN
from userStates import UserStates
from userStates import PairInput
import texts as tx

bot = Bot(token=TOKEN)
dp = Dispatcher(storage=MemoryStorage())

router = Router()
pair = {}
pair_predict = {}

main_keyboard1 = ReplyKeyboardMarkup(
    keyboard= [[KeyboardButton(text="Составить портфель"), KeyboardButton(text="Сделать прогноз")],
    ],
    resize_keyboard=True
)

main_keyboard2 = ReplyKeyboardMarkup(
    keyboard= [[KeyboardButton(text="Стоп"), KeyboardButton(text="Очистить список")],
                [KeyboardButton(text="Главное меню")]
    ],
    resize_keyboard=True
)


@router.message(CommandStart())
async def cmd_start(message: types.Message):
    await message.answer(
        tx.start_command(),
        reply_markup=main_keyboard1)


@router.message(F.text =="Главное меню")
async def contacts(message: types.Message):
    await message.answer(tx.return_main(), reply_markup=main_keyboard1)


@router.message(F.text =="Составить портфель")
async def contacts(message: types.Message, state: FSMContext):
    await state.set_state(PairInput.waiting_for_pairs)
    await message.answer(tx.start_make_portfolio(), reply_markup=main_keyboard2)

@router.message(F.text =="Сделать прогноз")
async def contacts(message: types.Message, state: FSMContext):
    await state.set_state(PairInput.predict_pairs)
    await message.answer(tx.start_make_predict(), reply_markup=main_keyboard2)


@router.message(F.text =="Очистить список")
async def contacts(message: types.Message):
    pair[message.from_user.id] = []
    await message.answer("Список очищен")


@router.message(StateFilter(PairInput.predict_pairs))
async def add_pair(message: types.Message, state: FSMContext):
    if message.text.lower() == "стоп":
        if message.from_user.id in pair_predict and pair_predict[message.from_user.id]:
            await message.answer("Ввод завершён. Ваши пары сохранены.")
            await message.answer("Введите количество дней, на которое вы хотите сделать прогноз(от 1 до 10 дней):")
            await state.set_state(UserStates.Predict)
        else:
            await message.answer("Пожалуйста, добавьте хотя бы одну пару перед завершением.")
        return
    pair_input = message.text.strip().upper()

    try:
        markets = exchange.load_markets()
        market_symbols = set(markets.keys())

        if pair_input not in market_symbols:
            await message.answer(f"Пара {pair_input} не найдена на бирже Bybit.")
            return

        if message.from_user.id not in pair_predict:
            pair_predict[message.from_user.id] = []

        if pair_input in pair_predict[message.from_user.id]:
            await message.answer(f"Пара {pair_input} уже есть в вашем списке.")
        else:
            pair_predict[message.from_user.id].append(pair_input)
            await message.answer(f"Пара {pair_input} добавлена в ваш список.")

        await message.answer(f"Ваш список: {pair_predict[message.from_user.id]}")

    except Exception as e:
        await message.answer(f"Ошибка при проверке монеты: {e}")

    if message.from_user.id not in pair_predict or not pair_predict[message.from_user.id]:
        await message.answer("Пожалуйста, введите первую пару.")

@router.message(StateFilter(UserStates.Predict))
async def predict_value(message: types.Message, state: FSMContext):
    # Проверка на ввод числа
    if not message.text.isdigit():
        await message.answer("Введите корректное число.")
        return

    forecast_horizon = int(message.text)

    # Проверка на диапазон суммы
    if 1 <= forecast_horizon <= 10:
        try:
            await message.answer("Дайте мне немного времени для составления прогноза.")
            # Получаем список пар пользователя из глобальной переменной
            user_pairs = pair_predict.get(message.from_user.id, [])

            # Создаем портфель на основе введённых пар
            results = data_forecast(user_pairs, forecast_horizon)
            await message.answer(tx.make_predict(results), reply_markup=main_keyboard1)
            pair_predict[message.from_user.id] = []
        except Exception as e:
            await message.answer("Произошла ошибка при составлении прогноза.")
            print(f"Error: {e}")
        finally:
            await state.clear()



@router.message(StateFilter(PairInput.waiting_for_pairs))
async def add_to_list(message: types.Message, state: FSMContext):
    if message.text.lower() == "стоп":
        if message.from_user.id in pair and pair[message.from_user.id]:
            await message.answer("Ввод завершён. Ваши пары сохранены.")
            await message.answer("Введите сумму для создания инвестиционного портфеля (от 1 000$ до 10 000 000$):")
            await state.set_state(UserStates.Money)
        else:
            await message.answer("Пожалуйста, добавьте хотя бы одну пару перед завершением.")
        return

    pair_input = message.text.strip().upper()

    try:
        markets = exchange.load_markets()
        market_symbols = set(markets.keys())

        if pair_input not in market_symbols:
            await message.answer(f"Пара {pair_input} не найдена на бирже Bybit.")
            return

        if message.from_user.id not in pair:
            pair[message.from_user.id] = []

        if pair_input in pair[message.from_user.id]:
            await message.answer(f"Пара {pair_input} уже есть в вашем списке.")
        else:
            pair[message.from_user.id].append(pair_input)
            await message.answer(f"Пара {pair_input} добавлена в ваш список.")

        await message.answer(f"Ваш список: {pair[message.from_user.id]}")

    except Exception as e:
        await message.answer(f"Ошибка при проверке монеты: {e}")

    if message.from_user.id not in pair or not pair[message.from_user.id]:
        await message.answer("Пожалуйста, введите первую пару.")

@router.message(StateFilter(UserStates.Money))
async def enter_amount(message: types.Message, state: FSMContext):
    # Проверка на ввод числа
    if not message.text.isdigit():
        await message.answer("Введите корректное число.")
        return

    amount = int(message.text)

    # Проверка на диапазон суммы
    if 1000 <= amount <= 10000000:
        try:
            await message.answer("Дайте мне немного времени для составления портфеля.")
            # Получаем список пар пользователя из глобальной переменной
            user_pairs = pair.get(message.from_user.id, [])


            # Создаем портфель на основе введённых пар
            portfolio = create_investment_portfolio(user_pairs, amount)
            await message.answer(tx.make_portfolio(portfolio), reply_markup=main_keyboard1)
            pair[message.from_user.id] = []
        except Exception as e:
            await message.answer("Произошла ошибка при создании портфеля.")
            print(f"Error: {e}")
        finally:
            await state.clear()
    else:
        await message.answer("Сумма должна быть в пределах от 1000 до 10000000.")


@router.message()
async def answer(message: types.Message):
    await message.answer('Простите, я не понимаю вас')


async def main():
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())