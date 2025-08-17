# make_c_array.py
tflite_path = r"./fm_48_distilled_out_manual/student_int8.tflite"
out_cc      = r"./fm_48_distilled_out_manual/model_data.cc"
var_name    = "g_model"

with open(tflite_path, "rb") as f:
    data = f.read()

with open(out_cc, "w", encoding="utf-8") as f:
    f.write('#include <cstdint>\n')
    f.write('#ifdef __has_attribute\n#if __has_attribute(aligned)\n#define ALN __attribute__((aligned(16)))\n#else\n#define ALN\n#endif\n#else\n#define ALN\n#endif\n\n')
    f.write('extern "C" {\n')
    f.write(f'const unsigned char {var_name}[] ALN = {{\n')
    for i, b in enumerate(data):
        if i % 12 == 0: f.write("  ")
        f.write(f"0x{b:02x}, ")
        if i % 12 == 11: f.write("\n")
    f.write('\n};\n')
    f.write(f'const unsigned int {var_name}_len = {len(data)};\n')
    f.write('}\n')

print("Wrote:", out_cc)
