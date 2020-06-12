# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 23:29:50 2020

@author: Pawan Prasad
"""

from gym.envs.registration import register

register(
    id='pdlwrld-v0',
    entry_point='gym_pdlwrld.envs:PdlwrldEnv',
)
'''
register(
    id='pdlwrld-extrahard-v0',
    entry_point='gym_pdlwrld.envs:PdlwrldExtraHardEnv',
)
'''
