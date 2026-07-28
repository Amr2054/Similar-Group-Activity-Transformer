"""Shared constants for the FIFA WC22 tracking/event extraction pipeline."""

TARGET_EVENT_TYPES = {'SH', 'CR'}   # FO dropped — rarely a standalone possessionEventType, see project notes

DEFAULT_SAMPLE_SIZE = 100
DEFAULT_PRE_BUFFER = 10
DEFAULT_POST_BUFFER = 3

FEATURE_COLS = ['x', 'y', 'z', 'speed', 'visibility', 'is_attacking', 'dist_to_goal']
NUM_AGENT_SLOTS = 23  # 1 ball + 11 home + 11 away

ROLE_HOME = 0
ROLE_AWAY = 1
ROLE_BALL = 2

DEFAULT_PITCH_LENGTH = 105.0
DEFAULT_PITCH_WIDTH = 68.0
DEFAULT_ATTACKING_DIRECTION = 'R'

PLAYER_MAX_SPEED = 12.0
BALL_MAX_SPEED = 35.0