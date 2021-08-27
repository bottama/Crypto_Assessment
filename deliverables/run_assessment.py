""" Crypto Finance, assessment 2 """

"""
Author: Matteo Bottacini, bottacinimatteo@gmail.com
Last update: August 26, 2021
"""

# import modules
from src.utils1 import *
from src.utils2 import *


# Task 1
df = task_1(local_folder='deliverables', data_folder='Candidate_folder',
            feature='price', weight='amount', N=100, time_interval='15min')

# Task 2
df = task2(markets=['ETH/USD', 'BTC/USD'], days=360, interval=1, rolling_days=30, local_folder='deliverables')

