#!/usr/bin/env python3
# fashion_mnist_48_cnn_fast_v3_compat.py
# - No tf.image.rotate / addons
# - No AdamW (uses Adam + L2 regularization)
# - Works on older TF/Keras where SparseCCE lacks label_smoothing

import os, json, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ----------------------------
# Config
# ----------------------------
IMG_SIZE   = 48
BATCH_SIZE = 128
EPOCHS     = 40            # EarlyStopping will cut it when done
VAL_COUNT  = 10_000
SEED       = 42
OUT_DIR    = "./fm_48_fast_out_v3"
os.makedirs(OUT_DIR, exist_ok=True)

# Reproducibility
np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)
AUTOTUNE = tf.data.AUTOTUNE

# ----------------------------
# Data: load & resize to 48x48
# ----------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

def resize_to_48(x: np.ndarray) -> np.ndarray:
    # (N,28,28) -> (N,48,48,1) float32 in [0,1]
    x = x.astype(np.float32) / 255.0
    x = np.expand_dims(x, -1)
    x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE), method="bilinear").numpy()
    return x

x_train = resize_to_48(x_train)
x_test  = resize_to_48(x_test)

# 50k train / 10k val split
x_val, y_val = x_train[-VAL_COUNT:], y_train[-VAL_COUNT:]
x_train, y_train = x_train[:-VAL_COUNT], y_train[:-VAL_COUNT]
num_classes = 10

# ----------------------------
# tf.data pipelines + mild aug (NO rotate dependency)
# ----------------------------
def aug_fn(img, label):
    # pad + random crop back to 48x48
    img = tf.image.resize_with_pad(img, IMG_SIZE + 4, IMG_SIZE + 4)
    img = tf.image.random_crop(img, size=(IMG_SIZE, IMG_SIZE, 1))
    # tiny photometric jitter
    img = tf.image.random_brightness(img, max_delta=0.05)
    img = tf.image.random_contrast(img, lower=0.95, upper=1.05)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label

train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(buffer_size=len(x_train), seed=SEED, reshuffle_each_iteration=True)
            .map(aug_fn, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE).cache().prefetch(AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((x_val, y_val))
          .batch(BATCH_SIZE).cache().prefetch(AUTOTUNE))

test_ds = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
           .batch(BATCH_SIZE).prefetch(AUTOTUNE))

# ----------------------------
# Loss: smoothed Sparse CCE that's compatible everywhere
# ----------------------------
def smooth_sparse_cce(num_classes: int, label_smoothing: float = 0.05):
    ls = float(label_smoothing)
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=num_classes)
        if ls > 0.0:
            y_true_oh = y_true_oh * (1.0 - ls) + ls / num_classes
        # use categorical CE on (one-hot vs probs)
        return tf.keras.losses.categorical_crossentropy(y_true_oh, y_pred)
    return loss_fn

# ----------------------------
# Model (≈ 2× params vs your original)
# BN after convs (no conv bias) + L2 reg (AdamW surrogate)
# ----------------------------
L2 = keras.regularizers.l2(1e-4)

def conv_bn_relu(x, filters, k=3):
    x = L.Conv2D(filters, k, padding="same", use_bias=False,
                 kernel_initializer="he_normal", kernel_regularizer=L2)(x)
    x = L.BatchNormalization()(x)
    return L.ReLU()(x)

def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=10):
    inp = L.Input(shape=input_shape)

    # Block 1 (wider)
    x = conv_bn_relu(inp, 24, 3)
    x = conv_bn_relu(x,   24, 3)
    x = L.MaxPooling2D()(x)          # 48 -> 24
    x = L.Dropout(0.10)(x)

    # Block 2
    x = conv_bn_relu(x,   48, 3)
    x = L.MaxPooling2D()(x)          # 24 -> 12
    x = L.Dropout(0.10)(x)

    # Depthwise + pointwise block
    x = L.DepthwiseConv2D(3, padding="same", use_bias=False,
                          depthwise_regularizer=L2)(x)
    x = L.BatchNormalization()(x); x = L.ReLU()(x)
    x = L.Conv2D(72, 1, use_bias=False, kernel_regularizer=L2)(x)
    x = L.BatchNormalization()(x); x = L.ReLU()(x)
    x = L.MaxPooling2D()(x)          # 12 -> 6

    x = L.GlobalAveragePooling2D()(x)
    x = L.Dense(96, activation="relu", kernel_regularizer=L2)(x)
    x = L.Dropout(0.30)(x)
    out = L.Dense(num_classes, activation="softmax", kernel_regularizer=L2)(x)
    return keras.Model(inp, out)

model = build_model()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss = smooth_sparse_cce(num_classes=num_classes, label_smoothing=0.05)

model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
model.summary()

# ----------------------------
# Callbacks / logging
# ----------------------------
csv_logger = CSVLogger(os.path.join(OUT_DIR, "training_log.csv"), append=False)
ckpt = ModelCheckpoint(
    os.path.join(OUT_DIR, "best.keras"),
    monitor="val_accuracy", mode="max",
    save_best_only=True, verbose=1,
)
es  = EarlyStopping(monitor="val_accuracy", mode="max", patience=6,
                    restore_best_weights=True, verbose=1)
rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                        min_lr=1e-5, verbose=1)

# ----------------------------
# Training
# ----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[csv_logger, ckpt, es, rlr],
    verbose=2,
)

# Save per-epoch metrics
with open(os.path.join(OUT_DIR, "history.json"), "w", encoding="utf-8") as f:
    json.dump(history.history, f, ensure_ascii=False, indent=2)

# ----------------------------
# Test evaluation
# ----------------------------
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"Test  loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

# ----------------------------
# Save weights to JSON + model JSON
# ----------------------------
arch_json_path    = os.path.join(OUT_DIR, "model_architecture.json")
weights_json_path = os.path.join(OUT_DIR, "weights.json")

# 1) architecture
with open(arch_json_path, "w", encoding="utf-8") as f:
    f.write(model.to_json())

# 2) weights (shape/dtype/flattened values)
weights_serialized = []
for arr in model.get_weights():
    weights_serialized.append({
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "values": arr.flatten().tolist(),
    })
with open(weights_json_path, "w", encoding="utf-8") as f:
    json.dump({
        "format": "keras-weights-list",
        "keras_version": tf.keras.__version__,
        "weights": weights_serialized,
    }, f, ensure_ascii=False)

print(f"\nSaved in {OUT_DIR}:")
print(" - training_log.csv")
print(" - history.json")
print(" - best.keras")
print(" - model_architecture.json")
print(" - weights.json")
print(f"Test metrics → loss: {test_loss:.4f}, acc: {test_acc:.4f}")
