from aiogram.fsm.state import StatesGroup, State


class UserStates(StatesGroup):
    Money = State()

class PairInput(StatesGroup):
    waiting_for_pairs = State()