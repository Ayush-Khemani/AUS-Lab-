import csv

input_csv = "outputs/centers.csv"
output_csv = "outputs/centers_valid.csv"

rows = []
with open(input_csv, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            cx, cy, cz = float(row["cx"]), float(row["cy"]), float(row["cz"])
            r = float(row["radius"])
            inliers = int(row["inliers"])
            # 筛选条件
            if abs(cx) < 5 and abs(cy) < 5 and abs(cz) < 5 and 0.24 < r < 0.26 and inliers >= 100:
                rows.append(row)
        except Exception:
            continue

print(f"有效帧数: {len(rows)}")
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"筛选后的结果已保存到 {output_csv}")
