import os
from yacs.config import CfgNode as CN

_C = CN()
_C.NAME = ''

# data parameters
_C.DATA = CN()
_C.DATA.START = -1.0
_C.DATA.END = 1.0
_C.DATA.TRAIN_SIZE = 10000
_C.DATA.TEST_SIZE = 1000
_C.DATA.BATCH_SIZE = 500

# model parameters
_C.MODEL = CN()
_C.MODEL.NAME = 'SimpleLinear'
_C.MODEL.IN_FEAT = 1
_C.MODEL.OUT_FEAT = 1
_C.MODEL.LAYERS = [100, 200, 100]
_C.MODEL.ACT = 'relu'
_C.MODEL.ISRES = False
_C.MODEL.ISASI = False

# train parameters
_C.TRAIN = CN()
_C.TRAIN.LR = 0.00001
_C.TRAIN.LR_STEP = 20
_C.TRAIN.LR_FACTOR = 0.5
_C.TRAIN.EPOCHS = 100
_C.TRAIN.OPT = 'Adam'

# results parameters
_C.RESULTS = CN()
# step
_C.RESULTS.PRINT_FREQ = 10
# epoch
_C.RESULTS.EVAL_FREQ = 10
_C.RESULTS.PLOT_HEAT = 20
_C.OUTPUT = 'results'


def update_config(config, args):
    config.defrost()
    
    # merge from config.yaml
    print('=> merge config from {}'.format(args.cfg))
    config.merge_from_file(args.cfg)
    config.freeze()
   
    # TODO
    # merge from specific arguments

    config.freeze()

    
def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config


