from custom_task.gsm8k_steering_exact import TASKS_TABLE as GSM8K_TASKS_TABLE
from custom_task.math_greedy_steering import TASKS_TABLE as MATH_GREEDY_TASKS_TABLE
from custom_task.math_stock_semantics import TASKS_TABLE as MATH_STOCK_TASKS_TABLE


TASKS_TABLE = [
    *GSM8K_TASKS_TABLE,
    *MATH_GREEDY_TASKS_TABLE,
    *MATH_STOCK_TASKS_TABLE,
]
