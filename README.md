# FMNIST 48×48 — Teacher ➜ Student Distillation (TFLite + C array)

**Purpose**
This repo shows how to train a compact CNN on **Fashion-MNIST (48×48)** as a **teacher**, distill it into a **small student**, quantize to **INT8**, and export a **C array** for embedded use (e.g., RIOT OS).

There are also **zip files with pictures of people**. Theoretically, you could use the same pipeline to make a small camera with a microcontroller that counts how many people are in a shot. However, since there are not that many pictures, you may need to change the pipeline a bit. A `count.json` is included with placeholders (`null`) for all picture labels.

---

## 📂 Repo Contents

* **`fashion_mnist_48_cnn_fast_v3_compat.py`**
  Teacher training script (≈2× params vs the minimal version).

* **`distill_fmnist_48_manual_v2.py`**
  Student training with **knowledge distillation (KD)**.

  * Combines cross-entropy and KD loss.
  * Converts to **INT8 TFLite** and exports a **C array**.
  * Teacher model is loaded automatically from:

    1. `./fm_48_fast_out_v3/best.keras`
    2. `./fm_48_fast_out_v2/best.keras`
    3. `./fm_48_fast_out/best.keras`

* **`make_c_array.py`**
  Standalone fallback: converts a `.tflite` model into a C array (`model_data.cc`).

* **`people_pics_*.zip`**
  Example dataset for people-counting.

  * Includes `count.json` (all labels `null` by default).
  * Can be annotated with the number of people per image and reused with a similar pipeline.

---

## 🚀 Quick Start

### 1. Train the Teacher

```bash
python fashion_mnist_48_cnn_fast_v3_compat.py
```

### 2. Distill into a Student

```bash
python distill_fmnist_48_manual_v2.py
```

Produces:

* `student_kd.keras`
* `student_int8.tflite`
* `model_data.cc`

### 3. (Optional fallback) Generate `model_data.cc` manually

If bugs occur during Step 2’s export:

```bash
python make_c_array.py --model student_int8.tflite --out model_data.cc
```

### 4. Deploy in RIOT OS

Copy `model_data.cc` into your RIOT project and link against TensorFlow Lite Micro.

---

## ⚡ Notes

* Teacher: \~89% accuracy on FMNIST (48×48)
* Student: \~81% accuracy (INT8, distilled)
* For people-counting, pipeline is theoretically the same, but dataset size may require tweaks.
