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

from aiogram.filters import StateFilter
from data import config
from data.config import TOKEN
from userStates import UserStates
from userStates import PairInput
import texts as tx

bot = Bot(token=TOKEN)
dp = Dispatcher(storage=MemoryStorage())

router = Router()
pair = {}

main_keyboard1 = ReplyKeyboardMarkup(
    keyboard= [[KeyboardButton(text="Составить портфель"), KeyboardButton(text="Сделать прогноз")],
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
    await message.answer(tx.start_make_portfolio())

@router.message(PairInput.waiting_for_pairs, lambda message: not message.text.startswith('/'))
async def add_to_list(message: types.Message):
    if message.text.startswith('/'):
        # Это команда, не обрабатываем как пару
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

@router.message(Command("done"), PairInput.waiting_for_pairs)
async def done(message: types.Message, state: FSMContext):
    user_id = message.from_user.id
    await state.clear()  # сбрасываем состояние — ввод завершён
    await message.answer("Ввод завершён. Ваши пары сохранены.")

@router.message(StateFilter(UserStates.Money))
async def answer(message: types.Message, state: FSMContext):
    if message.text.lower() == 'отмена':
        await message.answer(tx.start_make_portfolio())
        await state.clear()
    else:
        if '/' not in message.text:
            if message.text.isdigit():
                amount = int(message.text)
                if 1000 <= amount <= 10000000:
                    try:
                        portfolio = algos.create_investment_portfolio(algos.stocks, amount)
                        await message.answer(tx.make_portfolio(portfolio))
                    except Exception as e:
                        await message.answer("Произошла ошибка при обработке запроса.")
                        print(f"Error: {e}")
                    finally:
                        await state.clear()
                else:
                    await message.answer(tx.low_money())
            else:
                await message.answer(tx.is_not_digit())
        else:
            await message.answer(tx.bad_responce())
            await state.clear()


@router.message(F.text == "Дай пасхалку")
async def answer(message: types.Message):
    await message.answer('https://www.youtube.com/watch?v=dQw4w9WgXcQ')


@router.message()
async def answer(message: types.Message):
    await message.answer('Прости, я не понимаю тебя')


async def main():
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())