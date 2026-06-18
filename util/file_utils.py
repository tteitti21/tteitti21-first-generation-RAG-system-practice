import json
import os


def json_file_has_content(file_path):
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(file_path, data):
    parent_dir = os.path.dirname(file_path)

    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
