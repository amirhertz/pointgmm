import os


EPSILON = 1e-6
DIM = 3
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
DARA_ROOT =  f'{PROJECT_ROOT}/dataset'
DATASET = f'{DARA_ROOT}/ShapeNetCore.v2'
CHECKPOINTS_ROOT = f'{PROJECT_ROOT}/checkpoints'
REGISTRATION = f'{DARA_ROOT}/registration'
OUT_DIR = f'{PROJECT_ROOT}/generated'

