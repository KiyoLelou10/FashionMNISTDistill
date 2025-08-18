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
python3 FashionMnistCNN.py
```

### 2. Distill into a Student

```bash
python3 CNNDistillRIOT.py
```

Produces:

* `student_kd.keras`
* `student_int8.tflite`
* `model_data.cc`

### 3. (Optional fallback) Generate `model_data.cc` manually

If bugs occur during Step 2’s export:

```bash
python3 WeightsForRIOT.py
```

### 4. Deploy in RIOT OS

Copy `model_data.cc` into your RIOT project and link against TensorFlow Lite Micro.

---

## ⚡ Notes

* Teacher: \~89% accuracy on FMNIST (48×48)
* Student: \~85% accuracy (INT8, distilled)
* For people-counting, pipeline is theoretically the same, but dataset size may require tweaks.

🧮 Loss functions (teacher & student)
Teacher training (fashion_mnist_48_cnn_fast_v3_compat.py)

Objective: smoothed cross-entropy on 10-class Fashion-MNIST.

Loss: label-smoothed categorical cross-entropy (via one-hot)

𝑦
~
  
=
  
(
1
−
𝜀
)
 
o
n
e
h
o
t
(
𝑦
)
  
+
  
𝜀
𝐾
 
1
with 
𝜀
=
0.05
,
  
𝐾
=
10
y
~
	​

=(1−ε)onehot(y)+
K
ε
	​

1with ε=0.05,K=10
𝐿
teacher
  
=
  
C
E
 ⁣
(
𝑦
~
,
 
𝑝
𝜃
)
L
teacher
	​

=CE(
y
~
	​

,p
θ
	​

)

where $\mathbf{p}_{\theta}$ are the model softmax outputs.

In code: smooth_sparse_cce(num_classes=10, label_smoothing=0.05) → keras.losses.categorical_crossentropy

Optimizer: Adam (lr = 1e-3)

Regularization: L2 (1e-4) on conv/dense kernels

Augmentations: pad+random crop, light brightness/contrast jitter

Student distillation (distill_fmnist_48_manual_v2.py)

Objective: combine hard-label CE with a temperature-scaled KD term from the frozen teacher.

Hard-label term (student logits $\mathbf{z}_s$):

𝐿
CE
  
=
  
C
E
sparse
 ⁣
(
𝑦
,
 
𝑧
𝑠
)
L
CE
	​

=CE
sparse
	​

(y,z
s
	​

)

(implemented with SparseCategoricalCrossentropy(from_logits=True))

KD term (teacher probs $\mathbf{p}_t$):

𝑝
𝑡
(
𝑇
)
  
=
  
n
o
r
m
 ⁣
(
𝑝
𝑡
 
1
/
𝑇
)
,
𝑝
𝑠
(
𝑇
)
  
=
  
s
o
f
t
m
a
x
 ⁣
(
𝑧
𝑠
/
𝑇
)
p
t
(T)
	​

=norm(p
t
1/T
	​

),p
s
(T)
	​

=softmax(z
s
	​

/T)
𝐿
KD
  
=
  
𝑇
2
⋅
K
L
 ⁣
(
𝑝
𝑡
(
𝑇
)
 
∥
 
𝑝
𝑠
(
𝑇
)
)
L
KD
	​

=T
2
⋅KL(p
t
(T)
	​

	​

p
s
(T)
	​

)

Total loss:

𝐿
student
  
=
  
𝛼
 
𝐿
CE
  
+
  
(
1
−
𝛼
)
 
𝐿
KD
with 
𝛼
=
0.5
,
  
𝑇
=
3.0
L
student
	​

=αL
CE
	​

+(1−α)L
KD
	​

with α=0.5,T=3.0

Optimizer: Adam (lr = 1e-3)

Notes: Teacher outputs are coerced to probabilities (softmax applied if needed). KD uses the standard $T^2$ scaling.

Quantization (export path)

Full-integer INT8 conversion using a representative dataset (up to 500 samples), with inference_input_type = int8 and inference_output_type = int8.

Outputs:

student_kd.keras — Keras checkpoint

student_int8.tflite — quantized model

model_data.cpp — C array (g_model, g_model_len) for embedded targets (e.g., RIOT OS)
