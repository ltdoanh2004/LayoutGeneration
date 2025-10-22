import os
import json
import pandas as pd

# Path đến thư mục chứa các kết quả
base_dir = "outputs/outputs_eval"

# Danh sách để lưu kết quả
results = []

# Loop qua các folder con trong outputs_eval
for subdir, dirs, files in os.walk(base_dir):
    if "eval_results.json" in files:
        json_path = os.path.join(subdir, "eval_results.json")
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                data["folder"] = os.path.basename(subdir)  # thêm tên folder để theo dõi
                results.append(data)
        except Exception as e:
            print(f"Error reading {json_path}: {e}")

# Chuyển sang DataFrame để dễ xử lý
df = pd.DataFrame(results)

# Tính trung bình các metric (bỏ cột folder)
summary = df.drop(columns=["folder"]).mean().to_frame(name="Mean").T

print("=== Summary of Metrics Across All Evaluations ===")
print(summary)

# Xuất ra file CSV nếu muốn
summary.to_csv("outputs/summary_eval_metrics.csv", index=False)
