from pathlib import Path
from PIL import Image


def check_folder(folder):
    bad_files = []
    total = 0
    for class_dir in sorted(folder.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            total += 1
            try:
                with Image.open(img_path) as image:
                    image.verify()
            except Exception as error:
                bad_files.append((str(img_path), str(error)))
    return total, bad_files


train_total, train_bad = check_folder(Path('ml/data/merged/train'))
val_total, val_bad = check_folder(Path('ml/data/merged/val'))

print(f"Train: {train_total} images checked, {len(train_bad)} corrupt")
print(f"Val: {val_total} images checked, {len(val_bad)} corrupt")

if train_bad or val_bad:
    print("\nCORRUPT FILES:")
    for path, error in (train_bad + val_bad)[:20]:
        print(f"  {path}: {error}")
    if len(train_bad + val_bad) > 20:
        print(f"  ... and {len(train_bad + val_bad) - 20} more")
