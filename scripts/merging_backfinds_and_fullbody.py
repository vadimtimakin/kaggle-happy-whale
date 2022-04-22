import os
import shutil
from tqdm import tqdm

fullbody = "/home/toefl/K/dolphin/datasets/fullbody/test"
backfin = "/home/toefl/K/dolphin/datasets/backfins/test_images"

backfin_samples = os.listdir(backfin)
c = 0
for file in tqdm(os.listdir(fullbody)):
    if file not in backfin_samples:
        shutil.copy(os.path.join(fullbody, file), os.path.join(backfin, file))
        c += 1

print(c)