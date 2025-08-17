#!/usr/bin/env python3
# distill_fmnist_48_manual_v2.py
# - Loads the *new* teacher safely (compile=False), even if it was saved with a custom loss
# - Handles teacher outputs whether they're softmax probs or raw logits
# - Keeps your manual KD loop, TFLite INT8 conversion, and C array emitter

import os, json, random, math, time, csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
import pandas as pd

# ----------------------------
# Config
# ----------------------------
SEED = 42
IMG_SIZE = 48
BATCH_SIZE = 128
EPOCHS = 20              # distillation epochs
LR = 1e-3
ALPHA = 0.5             # weight for CE vs KD
TEMP = 3.0              # KD temperature
VAL_COUNT = 10_000
OUT_DIR = "./fm_48_distilled_out_manual"

# Prefer the newer teacher; fall back if needed
CANDIDATE_TEACHERS = [
    "./fm_48_fast_out_v3/best.keras",  # from the compat BN/L2 script
    "./fm_48_fast_out_v2/best.keras",  # from the v2 script
    "./fm_48_fast_out/best.keras",     # your original (last resort)
]

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

# ----------------------------
# Data: Fashion-MNIST → 48x48 grayscale [0,1]
# ----------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

def resize48(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32) / 255.0
    x = np.expand_dims(x, -1)
    x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE), method="bilinear").numpy()
    return x

x_train = resize48(x_train)
x_test  = resize48(x_test)

# 50k / 10k split from 60k train
x_val, y_val = x_train[-VAL_COUNT:], y_train[-VAL_COUNT:]
x_train, y_train = x_train[:-VAL_COUNT], y_train[:-VAL_COUNT]

NUM_CLASSES = 10

# ----------------------------
# Teacher (probabilities) — safe load + robust outputs
# ----------------------------
TEACHER_PATH = next((p for p in CANDIDATE_TEACHERS if os.path.exists(p)), None)
if TEACHER_PATH is None:
    raise SystemExit("No teacher checkpoint found. Checked:\n" + "\n".join(CANDIDATE_TEACHERS))
print("Using teacher:", TEACHER_PATH)

# Avoid deserializing any custom loss/optimizer
teacher = keras.models.load_model(TEACHER_PATH, compile=False)
teacher.trainable = False

def predict_probs(model, x, batch=256):
    """Return probabilities regardless of teacher’s final layer (softmax or logits)."""
    out = []
    for i in range(0, len(x), batch):
        xb = x[i:i+batch]
        yb = model.predict(xb, batch_size=batch, verbose=0)
        yb = tf.convert_to_tensor(yb, dtype=tf.float32)
        # Heuristic: if rows don't sum ~1, treat as logits
        row_sums = tf.reduce_sum(tf.nn.relu(yb), axis=-1)
        need_softmax = tf.reduce_any(tf.logical_or(tf.math.is_nan(row_sums), row_sums < 0.99))
        probs = tf.nn.softmax(yb) if bool(need_softmax.numpy()) else yb
        out.append(probs.numpy().astype(np.float32))
    return np.concatenate(out, axis=0)

print("Precomputing teacher probabilities...")
t_train = predict_probs(teacher, x_train)
t_val   = predict_probs(teacher, x_val)

# ----------------------------
# Student (TFLM-friendly)
# ----------------------------
def ds_block(x, c, s=1):
    x = L.DepthwiseConv2D(3, strides=s, padding="same", activation="relu")(x)
    x = L.Conv2D(c, 1, activation="relu")(x)
    return x

def build_student(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES):
    inp = L.Input(shape=input_shape)
    x = L.Conv2D(16, 3, strides=1, padding="same", activation="relu")(inp)
    x = ds_block(x, 24, s=2)     # 48->24
    x = ds_block(x, 32, s=1)
    x = ds_block(x, 48, s=2)     # 24->12
    x = ds_block(x, 64, s=1)
    x = L.AveragePooling2D(pool_size=3, strides=2, padding="same")(x)  # ~12->6
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dense(64, activation="relu")(x)
    out = L.Dense(num_classes)(x)   # logits
    return keras.Model(inp, out, name="student_small")

student = build_student()
optimizer = keras.optimizers.Adam(LR)

# Loss helpers
ce = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def kd_term(teacher_probs, student_logits, T=3.0):
    """
    KL( p_t^T || p_s^T ) * T^2, where
      p_t^T = normalize(p_t ** (1/T))
      p_s^T = softmax(logits / T)
    """
    eps = 1e-8
    T = tf.cast(T, tf.float32)
    p_t = tf.clip_by_value(teacher_probs, eps, 1.0)
    p_tT = tf.pow(p_t, 1.0 / T)
    p_tT = p_tT / tf.reduce_sum(p_tT, axis=-1, keepdims=True)
    p_sT = tf.nn.softmax(student_logits / T)
    kl = tf.reduce_sum(p_tT * (tf.math.log(p_tT + eps) - tf.math.log(p_sT + eps)), axis=-1)
    return tf.reduce_mean(kl) * (T * T)

# ----------------------------
# Training loop (manual KD)
# ----------------------------
def iterate_minibatches(X, y, T_probs, batch):
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    for i in range(0, len(X), batch):
        sel = idx[i:i+batch]
        yield X[sel], y[sel], T_probs[sel]

history_rows = []

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    # ---- Train ----
    train_loss = train_ce = train_kd = 0.0
    train_correct = 0
    n_train = 0

    for xb, yb, tb in iterate_minibatches(x_train, y_train, t_train, BATCH_SIZE):
        with tf.GradientTape() as tape:
            logits = student(xb, training=True)
            loss_ce = ce(yb, logits)
            loss_kd = kd_term(tb, logits, T=TEMP)
            loss = ALPHA * loss_ce + (1.0 - ALPHA) * loss_kd

        grads = tape.gradient(loss, student.trainable_variables)
        optimizer.apply_gradients(zip(grads, student.trainable_variables))

        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        train_correct += int(tf.reduce_sum(tf.cast(preds == tf.cast(yb, tf.int32), tf.int32)))
        bs = xb.shape[0]
        n_train += bs
        train_loss += float(loss) * bs
        train_ce   += float(loss_ce) * bs
        train_kd   += float(loss_kd) * bs

    train_loss /= n_train
    train_ce   /= n_train
    train_kd   /= n_train
    train_acc   = train_correct / n_train

    # ---- Validate ----
    val_loss = val_ce = 0.0
    val_correct = 0
    n_val = 0
    for i in range(0, len(x_val), BATCH_SIZE):
        xb = x_val[i:i+BATCH_SIZE]
        yb = y_val[i:i+BATCH_SIZE]
        logits = student(xb, training=False)
        loss_ce = ce(yb, logits)
        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        val_correct += int(tf.reduce_sum(tf.cast(preds == tf.cast(yb, tf.int32), tf.int32)))
        bs = xb.shape[0]
        n_val += bs
        val_ce += float(loss_ce) * bs
    val_loss = val_ce / n_val
    val_acc  = val_correct / n_val

    dt = time.time() - t0
    row = {
        "epoch": epoch,
        "time_sec": round(dt, 2),
        "train_loss": round(train_loss, 6),
        "train_ce": round(train_ce, 6),
        "train_kd": round(train_kd, 6),
        "train_accuracy": round(train_acc, 6),
        "val_loss": round(val_loss, 6),
        "val_accuracy": round(val_acc, 6),
        "lr": float(optimizer.learning_rate.numpy()),
    }
    print(f"Epoch {epoch:02d} | "
          f"loss {row['train_loss']:.4f} (ce {row['train_ce']:.4f}, kd {row['train_kd']:.4f}) "
          f"acc {row['train_accuracy']:.4f} | "
          f"val_loss {row['val_loss']:.4f} val_acc {row['val_accuracy']:.4f} "
          f"| {row['time_sec']}s")
    history_rows.append(row)

# Save history
hist_json = os.path.join(OUT_DIR, "history.json")
with open(hist_json, "w", encoding="utf-8") as f:
    json.dump(history_rows, f, ensure_ascii=False, indent=2)
pd.DataFrame(history_rows).to_csv(os.path.join(OUT_DIR, "training_log.csv"), index=False)

# Save student checkpoint
student_path = os.path.join(OUT_DIR, "student_kd.keras")
student.save(student_path)
print("Saved student:", student_path)

# ----------------------------
# INT8 TFLite conversion (full-integer)
# ----------------------------
def rep_ds():
    for i in range(0, min(500, len(x_train))):
        yield [x_train[i:i+1]]

converter = tf.lite.TFLiteConverter.from_keras_model(student)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_ds
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8
tflite_int8 = converter.convert()

tflite_path = os.path.join(OUT_DIR, "student_int8.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_int8)

# ----------------------------
# Evaluate INT8 TFLite
# ----------------------------
def eval_tflite_int8(model_bytes, X, Y, batch=128):
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    in_d  = interpreter.get_input_details()[0]
    out_d = interpreter.get_output_details()[0]

    sig = in_d.get("shape_signature", None)
    wanted = sig if sig is not None else in_d["shape"]
    wanted = list(map(int, wanted))
    can_resize = (wanted[0] == -1)  # dynamic batch

    if can_resize:
        new_shape = [batch] + wanted[1:]
        interpreter.resize_tensor_input(in_d["index"], new_shape, strict=False)
        interpreter.allocate_tensors()
        in_d  = interpreter.get_input_details()[0]
        out_d = interpreter.get_output_details()[0]

    in_scale, in_zp = in_d["quantization"]
    correct = total = 0

    if can_resize:
        for i in range(0, len(X), batch):
            xb = X[i:i+batch]
            yb = Y[i:i+batch]
            xq = np.round(xb / in_scale + in_zp).astype(np.int8)
            xq = np.clip(xq, -128, 127)
            interpreter.set_tensor(in_d["index"], xq)
            interpreter.invoke()
            logits_q = interpreter.get_tensor(out_d["index"])  # int8 logits [B,10]
            preds = logits_q.argmax(axis=1)
            correct += int((preds == yb).sum())
            total += len(yb)
    else:
        for i in range(len(X)):
            xb = X[i:i+1]
            yb = Y[i:i+1]
            xq = np.round(xb / in_scale + in_zp).astype(np.int8)
            xq = np.clip(xq, -128, 127)
            interpreter.set_tensor(in_d["index"], xq)
            interpreter.invoke()
            logits_q = interpreter.get_tensor(out_d["index"])  # [1,10]
            pred = int(logits_q[0].argmax())
            correct += int(pred == int(yb[0]))
            total += 1

    return correct / total

test_acc = eval_tflite_int8(tflite_int8, x_test, y_test)
print(f"TFLite INT8 test accuracy: {test_acc:.4f}")

# ----------------------------
# Emit C array for RIOT (aligned)
# ----------------------------
def write_c_array(tflite_bytes: bytes, out_cc: str, var_name="g_model"):
    with open(out_cc, "w", encoding="utf-8") as f:
        f.write('#include <cstdint>\n')
        f.write('#ifdef __has_attribute\n')
        f.write('#if __has_attribute(aligned)\n')
        f.write('#define ALN __attribute__((aligned(16)))\n')
        f.write('#else\n#define ALN\n#endif\n')
        f.write('#else\n#define ALN\n#endif\n\n')
        f.write('extern "C" {\n')
        f.write(f'const unsigned char {var_name}[] ALN = {{\n')
        for i, b in enumerate(tflite_bytes):
            if i % 12 == 0: f.write("  ")
            f.write(f"0x{b:02x}, ")
            if i % 12 == 11: f.write("\n")
        f.write('\n};\n')
        f.write(f'const unsigned int {var_name}_len = {len(tflite_bytes)};\n')
        f.write('}\n')

cc_path = os.path.join(OUT_DIR, "model_data.cc")
write_c_array(tflite_int8, cc_path)
print("Saved:", tflite_path, "and", cc_path)
