"""
Train a binary CNN classifier: tumor vs. no-tumor, from MRI images.

Expects a directory structure like:
    data/
        tumor/*.jpg
        no_tumor/*.jpg   (or "normal/*.jpg")

Usage:
    python train_model.py --data_dir data --epochs 15
"""
import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(img_size):
    model = models.Sequential(
        [
            layers.Input(shape=(img_size[0], img_size[1], 3)),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main(data_dir: str, output_dir: str = "models", epochs: int = 15):
    os.makedirs(output_dir, exist_ok=True)
    img_size = (128, 128)
    batch_size = 32

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
    )

    class_names = train_ds.class_names
    print(f"Class names (index order matters!): {class_names}")

    # ---- Compute class weights to correct for imbalance ----
    labels = []
    for _, y in train_ds.unbatch():
        labels.append(int(y.numpy()))
    labels = np.array(labels)
    class_counts = np.bincount(labels)
    print(f"Training image counts per class {class_names}: {class_counts.tolist()}")

    total = class_counts.sum()
    class_weight = {
        i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)
    }
    print(f"Using class weights: {class_weight}")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    model = build_model(img_size)
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stop],
        class_weight=class_weight,
    )

    val_loss, val_acc = model.evaluate(val_ds)
    print(f"Validation accuracy: {val_acc:.4f} | Validation loss: {val_loss:.4f}")

    model.save(os.path.join(output_dir, "tumor_model.h5"))
    with open(os.path.join(output_dir, "class_names.txt"), "w") as f:
        f.write("\n".join(class_names))
    print(f"Saved model + class names to '{output_dir}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="models")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.epochs)

    
# """
# Train a binary CNN classifier: tumor vs. no-tumor, from MRI images.

# Expects a directory structure like:
#     data/
#         tumor/*.jpg
#         no_tumor/*.jpg   (or "normal/*.jpg")

# Usage:
#     python train_model.py --data_dir data --epochs 15
# """
# import argparse
# import os

# import tensorflow as tf
# from tensorflow.keras import layers, models


# def build_model(img_size):
#     model = models.Sequential(
#         [
#             layers.Input(shape=(img_size[0], img_size[1], 3)),
#             layers.Rescaling(1.0 / 255),
#             layers.Conv2D(32, 3, activation="relu"),
#             layers.MaxPooling2D(),
#             layers.Conv2D(64, 3, activation="relu"),
#             layers.MaxPooling2D(),
#             layers.Conv2D(128, 3, activation="relu"),
#             layers.MaxPooling2D(),
#             layers.Flatten(),
#             layers.Dense(128, activation="relu"),
#             layers.Dropout(0.3),
#             layers.Dense(1, activation="sigmoid"),
#         ]
#     )
#     model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#     return model


# def main(data_dir: str, output_dir: str = "models", epochs: int = 15):
#     os.makedirs(output_dir, exist_ok=True)
#     img_size = (128, 128)
#     batch_size = 32

#     train_ds = tf.keras.utils.image_dataset_from_directory(
#         data_dir,
#         validation_split=0.2,
#         subset="training",
#         seed=123,
#         image_size=img_size,
#         batch_size=batch_size,
#         label_mode="binary",
#     )
#     val_ds = tf.keras.utils.image_dataset_from_directory(
#         data_dir,
#         validation_split=0.2,
#         subset="validation",
#         seed=123,
#         image_size=img_size,
#         batch_size=batch_size,
#         label_mode="binary",
#     )

#     class_names = train_ds.class_names
#     print(f"Class names (index order matters!): {class_names}")

#     autotune = tf.data.AUTOTUNE
#     train_ds = train_ds.cache().shuffle(1000).prefetch(autotune)
#     val_ds = val_ds.cache().prefetch(autotune)

#     model = build_model(img_size)
#     model.summary()

#     early_stop = tf.keras.callbacks.EarlyStopping(
#         monitor="val_loss", patience=5, restore_best_weights=True
#     )

#     history = model.fit(
#         train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early_stop]
#     )

#     val_loss, val_acc = model.evaluate(val_ds)
#     print(f"Validation accuracy: {val_acc:.4f} | Validation loss: {val_loss:.4f}")

#     model.save(os.path.join(output_dir, "tumor_model.h5"))
#     with open(os.path.join(output_dir, "class_names.txt"), "w") as f:
#         f.write("\n".join(class_names))
#     print(f"Saved model + class names to '{output_dir}/'")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_dir", required=True)
#     parser.add_argument("--output_dir", default="models")
#     parser.add_argument("--epochs", type=int, default=15)
#     args = parser.parse_args()
#     main(args.data_dir, args.output_dir, args.epochs)
