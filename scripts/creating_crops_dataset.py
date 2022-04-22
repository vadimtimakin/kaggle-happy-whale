import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

path_to_old = "/home/toefl/K/dolphin/happy-whale-and-dolphin/train_images"
path_to_new = "/home/toefl/K/dolphin/happy-whale-and-dolphin/train_crops"

df = pd.read_csv("/home/toefl/K/dolphin/happy-whale-and-dolphin/fullbody_train.csv")

src_bboxes = df["bbox"].values
bboxes = []
for bb in src_bboxes:
    bb = bb.split()
    if len(bb) == 5:
        bb = bb[1:]
    elif len(bb) == 4:
        bb[0] = bb[0][2:]
    bb[-1] = bb[-1][:-2]
    bb = [int(i) for i in bb]
    bboxes.append(bb)

files = df["image_id"].values

for file, bbox in zip(files, bboxes):
    img = cv2.imread(os.path.join(path_to_old, file))
    plt.imshow(img)
    plt.show()
    x, y, w, h = bbox
    img = img[y:y+h, x:x+w]
    plt.imshow(img)
    plt.show()