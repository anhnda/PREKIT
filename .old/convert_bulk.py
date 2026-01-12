import json

# Bước 1: Đọc dữ liệu từ file gốc
with open("train_fold_0_part_0.json", "r") as f:
    data = json.load(f)

bulk_lines = []

# Bước 2: Xử lý từng bản ghi
for item in data:
    measures = item.get("measures", {})
    for test_name in measures:
        test_values = measures[test_name]
        # Nếu là dict (timestamp: value), chuyển thành danh sách
        if isinstance(test_values, dict):
            measures[test_name] = [
                {"timestamp": ts, "value": val}
                for ts, val in test_values.items()
            ]
    # Bước 3: Thêm lệnh index và dữ liệu vào bulk_lines
    bulk_lines.append(json.dumps({ "index": { "_index": "icu-index" } }))
    bulk_lines.append(json.dumps(item))

# Bước 4: Ghi ra file bulk.json (ndjson format)
with open("bulk.json", "w") as f:
    for line in bulk_lines:
        f.write(line + "\n")

print("✅ Đã tạo file bulk.json sẵn sàng để gửi vào Elasticsearch.")
