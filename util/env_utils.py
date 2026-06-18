import os


def load_env_file(file_path):
    env_values = {}

    if not os.path.exists(file_path):
        return env_values

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            env_values[key.strip()] = value.strip().strip("\"'")

    return env_values


def get_env_path(env_values, key, base_dir):
    value = os.environ.get(key) or env_values.get(key)

    if not value:
        raise ValueError(f"Missing required path in .env: {key}")

    if os.path.isabs(value):
        return value

    return os.path.join(base_dir, value)
