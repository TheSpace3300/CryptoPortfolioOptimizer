try:
    markets = exchange.load_markets()
    market_symbols = set(markets.keys())

    coin_exists = any(coin in symbol for symbol in markets.keys())

    if coin_exists:
        if message.from_user.id not in pair:
            pair[message.from_user.id] = []

        if coin not in pair[message.from_user.id]:
            pair[message.from_user.id].append(coin)
            await message.answer(f"Монета {coin} найдена на бирже Bybit и добавлена в ваш список.")
        else:
            await message.answer(f"Монета {coin} уже есть в вашем списке.")

        await message.answer(f"Ваш список: {pair[message.from_user.id]}")
    else:
        await message.answer(f"Монета {coin} не найдена на Bybit.")
except Exception as e:
    await message.answer(f"Ошибка при проверке монеты: {e}")