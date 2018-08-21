import numpy as np
from pysc2.lib import actions
from pysc2.lib.features import SCREEN_FEATURES
import random

_PLAYER_RELATIVE = SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = SCREEN_FEATURES.unit_type.index
_SELECTED = SCREEN_FEATURES.selected.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_KD8CHARGE = actions.FUNCTIONS.Effect_KD8Charge_screen.id
_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]
_UP_STRIDE = 1
_DOWN_STRIDE = 1
_LEFT_STRIDE = 1
_RIGHT_STRIDE = 1

def wait(obs, env):
    action = [actions.FunctionCall(_NO_OP, [])]
    obs = env.step(action)[0]
    return obs

def KD8Charge(obs, env):
    player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
    army_y, army_x = (player_relative == _PLAYER_HOSTILE).nonzero()
    if not army_y.any():
        action = [actions.FunctionCall(_NO_OP, [])]
        obs = env.step(action)[0]
        if obs.last():
            return False
        else:
            return obs
    elif _KD8CHARGE in obs.observation["available_actions"]:
        index = np.argmax(army_y)
        target = [army_x[index], army_y[index]]
        action = [actions.FunctionCall(_KD8CHARGE, [_NOT_QUEUED, target])]
        obs = env.step(action)[0]
        if obs.last():
            return False
    return obs

def attack(obs, env):
    player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
    army_y, army_x = (player_relative == _PLAYER_HOSTILE).nonzero()
    if not army_y.any():
        action = [actions.FunctionCall(_NO_OP, [])]
        obs = env.step(action)[0]
        if obs.last():
            return False
        else:
            return obs
    elif _ATTACK_SCREEN in obs.observation["available_actions"]:
        index = np.argmax(army_y)
        target = [army_x[index], army_y[index]]
        action = [actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target])]
        obs = env.step(action)[0]
        if obs.last():
            return False
    return obs
'''
def moveUp(obs,env,stride=_UP_STRIDE):
    selected = obs.observation["screen"][_SELECTED]
    selected_y,selected_x=(selected == 1).nonzero()
    if not selected_y.any():
        action = [actions.FunctionCall(_NO_OP, [])]
        obs = env.step(action)[0]
        if obs.last():
            return False
        else:
            return obs
    elif(selected_y[0]<40-stride):
        target = [selected_x[0], selected_y[0]+stride]
        action = [actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])]
        obs = env.step(action)[0]
        if obs.last():
            return False
    else:
        action = [actions.FunctionCall(_NO_OP, [])]
        obs = env.step(action)[0]
        if obs.last():
            return False
        else:
            return obs
    return obs

def moveDown(obs,env,stride=_DOWN_STRIDE):
    selected = obs.observation["screen"][_SELECTED]
    selected_y, selected_x = (selected == 1).nonzero()
    if not selected_y.any():
        action = [actions.FunctionCall(_NO_OP, [])]
        obs = env.step(action)[0]
        if obs.last():
            return False
        else:
            return obs
    elif (selected_y[0] >stride):
        target = [selected_x[0], selected_y[0] - stride]
        action = [actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])]
        obs = env.step(action)[0]
        if obs.last():
            return False
    else:
        action = [actions.FunctionCall(_NO_OP, [])]
        obs = env.step(action)[0]
        if obs.last():
            return False
        else:
            return obs
    return obs

def moveLeft(obs,env,stride=_LEFT_STRIDE):
    selected = obs.observation["screen"][_SELECTED]
    selected_y, selected_x = (selected == 1).nonzero()
    if not selected_y.any():
        action = [actions.FunctionCall(_NO_OP, [])]
        obs = env.step(action)[0]
        if obs.last():
            return False
        else:
            return obs
    elif (selected_x[0] >stride):
        target = [selected_x[0]-stride, selected_y[0]]
        action = [actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])]
        obs = env.step(action)[0]
        if obs.last():
            return False
    else:
        action = [actions.FunctionCall(_NO_OP, [])]
        obs = env.step(action)[0]
        if obs.last():
            return False
        else:
            return obs
    return obs

def moveRight(obs,env,stride=_RIGHT_STRIDE):
    selected = obs.observation["screen"][_SELECTED]
    selected_y, selected_x = (selected == 1).nonzero()
    if not selected_y.any():
        action = [actions.FunctionCall(_NO_OP, [])]
        obs = env.step(action)[0]
        if obs.last():
            return False
        else:
            return obs
    elif (selected_x[0] < 40-stride):
        target = [selected_x[0]+stride, selected_y[0]]
        action = [actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])]
        obs = env.step(action)[0]
        if obs.last():
            return False
    else:
        action = [actions.FunctionCall(_NO_OP, [])]
        obs = env.step(action)[0]
        if obs.last():
            return False
        else:
            return obs
    return obs
'''
def get_action(action, obs, env):
    combined_action = [wait, KD8Charge, attack]
    action = action.tolist()
    action_id = action.index(max(action))
    obs = combined_action[action_id](obs, env)
    return obs