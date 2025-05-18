from aiogram.fsm.state import StatesGroup, State


class UserStates(StatesGroup):
    Money = State()
    Predict = State()

class PairInput(StatesGroup):
    waiting_for_pairs = State()
    predict_pairs = State()