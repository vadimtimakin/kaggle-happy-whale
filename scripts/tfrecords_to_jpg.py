import re
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

SRC_DIR = Path("/") / "home/toefl/K/dolphin"
HAPPY_WHALE_AND_DOLPHIN_DIR = SRC_DIR / "happy-whale-and-dolphin"
BACKFINTFRECORDS_DIR = SRC_DIR / "tfrecords"
HAPPY_WHALE_AND_DOLPHIN_BACKFIN_DIR = SRC_DIR / "backfins"


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


def load_dataset(filenames, image_size):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda x: read_labeled_tfrecord(x, image_size))
    return dataset


def read_labeled_tfrecord(example, image_size):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example["image"], image_size)
    image_name = example["image_name"]
    target = example["target"]

    return image, image_name, target


def decode_image(image_data, image_size):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    # image = tf.image.resize(image, image_size)
    return image


def convert_dataset(step, image_size):
    print(f"Converting {step} backfintfrecords with image size {image_size}")

    backfin_images_dir = HAPPY_WHALE_AND_DOLPHIN_BACKFIN_DIR / f"{step}_images"
    backfin_images_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created {backfin_images_dir}")

    filenames = tf.io.gfile.glob(f"{BACKFINTFRECORDS_DIR}/happywhale-2022-{step}*.tfrec")
    print(f"Number of {step} tfrecords: {len(filenames)}")

    num_items = count_data_items(filenames)
    print(f"Number of {step} images: {num_items}")

    dataset = load_dataset(filenames, image_size)

    image_names = []
    for sample in tqdm(dataset, total=num_items, desc=f"Saving {step} images"):
        image, image_name, _ = sample
        image, image_name = image.numpy(), image_name.numpy().decode("utf-8")

        image = (image * 255.0).astype(np.uint8)

        image = Image.fromarray(image)
        image.save(backfin_images_dir / image_name)

        image_names.append(image_name)

    df_backfin = pd.DataFrame({"image": image_names})
    
    filename = f"{step}.csv" if step == "train" else "sample_submission.csv"

    # Remove missing images from original df
    df_original = pd.read_csv(HAPPY_WHALE_AND_DOLPHIN_DIR / filename)
    df_backfin = pd.merge(df_original, df_backfin, how="inner", on="image")

    df_backfin.to_csv(HAPPY_WHALE_AND_DOLPHIN_BACKFIN_DIR / filename, index=False)
    
    print(f"Created {HAPPY_WHALE_AND_DOLPHIN_BACKFIN_DIR / filename}")


convert_dataset("train", (512, 512))
convert_dataset("test", (512, 512))