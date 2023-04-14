
from gymnasium.envs.registration import register

register(
    id="GettexStocks-v0",
    entry_point="gym_gettex.envs:GettexStocksEnv")