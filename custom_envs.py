from gymnasium.envs.registration import register
from minigrid.envs.redbluedoors import RedBlueDoorEnv

register(
    id='MiniGrid-RedBlueDoors-10x10-v0',
    entry_point='minigrid.envs.redbluedoors:RedBlueDoorEnv',
    kwargs={'size': 10}
)

register(
    id='MiniGrid-RedBlueDoors-12x12-v0',
    entry_point='minigrid.envs.redbluedoors:RedBlueDoorEnv',
    kwargs={'size': 12}
)

register(
    id='MiniGrid-RedBlueDoors-7x7-v0',
    entry_point='minigrid.envs.redbluedoors:RedBlueDoorEnv',
    kwargs={'size': 7}
)