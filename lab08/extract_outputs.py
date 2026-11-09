"""Lấy các file kết quả được nhúng trong experiments.ipynb (chạy Colab qua VS Code).

Cách dùng: python extract_outputs.py [đường_dẫn_notebook] [thư_mục_đích]
"""
import base64
import json
import os
import sys

MIME = "application/x-lab08-files+json"

nb_path = sys.argv[1] if len(sys.argv) > 1 else "experiments.ipynb"
out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(nb_path))

with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

payload = None
for cell in nb["cells"]:
    for out in cell.get("outputs", []):
        data = out.get("data", {})
        if MIME in data:
            payload = data[MIME]
            if isinstance(payload, str):
                payload = json.loads(payload)

if not payload:
    sys.exit("Không tìm thấy file nhúng. Hãy chạy cell cuối trên Colab rồi lưu notebook (Ctrl+S).")

os.makedirs(out_dir, exist_ok=True)
for name, b64 in payload.items():
    path = os.path.join(out_dir, os.path.basename(name))
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))
    print(f"Đã ghi {path} ({os.path.getsize(path):,} bytes)")
