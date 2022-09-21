import os
import logging
from datetime import datetime

from termcolor import colored

def setup_logging(args, config):
    run_str = datetime.now().strftime('%y_%m_%d_%H_%M_%S_{}'.format(config.MODEL.NAME))
    if args.tag is not None:
        run_str += f'_{args.tag}'
    run_dir = os.path.join(config.OUTPUT, run_str)
    
    config.defrost()
    config.OUTPUT = run_dir
    config.RESULTS.LOGS_DIR = os.path.join(run_dir, 'logs')
    config.RESULTS.CKPT_DIR = os.path.join(run_dir, 'checkPoints')
    config.freeze()
    os.makedirs(config.RESULTS.LOGS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS.CKPT_DIR, exist_ok=True)
    
    log_file = 'log.txt'
    final_log_file = os.path.join(run_dir, log_file)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s][%(name)s]:%(message)s')

    color_fmt = colored('[%(asctime)s]', 'green') + \
                colored('[%(name)s]', 'yellow') + \
                colored(': %(message)s', 'red')
    
    # create streamhandler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt=color_fmt))
    logger.addHandler(ch)

    # create filehandler
    sh = logging.FileHandler(str(final_log_file))
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
