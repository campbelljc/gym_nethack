import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='NetHackCombat-v0',
    entry_point='gym_nethack.envs:NetHackCombatEnv',
    reward_threshold=1.0,
    nondeterministic = True,
)

register(
    id='NetHackExplEnv-v0',
    entry_point='gym_nethack.envs:NetHackExplEnv',
    reward_threshold=1.0,
    nondeterministic = True,
)

register(
    id='NetHackLevel-v0',
    entry_point='gym_nethack.envs:NetHackLevelEnv',
    reward_threshold=1.0,
    nondeterministic = True,
)
