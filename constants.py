import os


EPSILON = 1e-6
DIM = 3
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
DARA_ROOT =  f'{PROJECT_ROOT}/dataset'
DATASET = f'{DARA_ROOT}/ShapeNetCoreParsed'
CHECKPOINTS_ROOT = f'{PROJECT_ROOT}/checkpoints'
REGISTRATION = f'{DARA_ROOT}/registration'
REG_OUT = f'{PROJECT_ROOT}/registration'
ART_DIR = f'{PROJECT_ROOT}/generated'

