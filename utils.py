import csv


def save_to_csv(headers: list, data: list, file_path: str):
    with open(file_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)
