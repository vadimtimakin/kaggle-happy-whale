import sys
import os

from utils import set_seed
from train_functions import run, run_eval
from config import config

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    set_seed(config.seed)

    if len(sys.argv) > 1:
        run_eval(config, checkpoint_path=sys.argv[1], fold=int(sys.argv[2]))

    else:
        if not os.path.exists(config.paths.save_dir):
            os.makedirs(config.paths.save_dir, exist_ok=True)

        n_folds = config.data.n_folds
        src_path = config.paths.save_dir
        scores = []
        final_score = 0

        for fold in [*range(n_folds)]:
            print(f'Fold: {fold}')
            config.paths.save_dir = os.path.join(src_path, f'fold_{fold}')
            cur_score = run(config, fold)
            final_score += cur_score / n_folds
            scores.append(cur_score)

        for fold in range(n_folds):
            print(f'Fold: {fold} | Score: {scores[fold]}')