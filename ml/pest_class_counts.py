"""Per-class image counts for IP102, so a scoped subset can be chosen on evidence.

Reads ml/data/pest/ip102/classification/{train,val,test}/<0..101>/ and joins the counts to
classes.txt, which is 1-indexed: folder N corresponds to classes.txt line N+1. That offset is
made explicit in the output because Ultralytics emits class indices 0..N-1 in folder order.

Writes ml/_pest_class_counts.tsv (scratch, gitignored via ml/_*).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
IP102 = ROOT / "data" / "pest" / "ip102"
CLASSES = IP102 / "classification"
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def main() -> int:
    names = [
        line.strip() for line in (IP102 / "classes.txt").read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    # strip the leading "N  " index that classes.txt carries
    cleaned = []
    for line in names:
        parts = line.split(None, 1)
        cleaned.append(parts[1].strip() if len(parts) == 2 and parts[0].isdigit() else line)

    folders = sorted(
        (d for d in (CLASSES / "train").iterdir() if d.is_dir()),
        key=lambda p: int(p.name) if p.name.isdigit() else 10**9,
    )

    rows = []
    for idx, folder in enumerate(folders):
        counts = {}
        for split in SPLITS:
            split_dir = CLASSES / split / folder.name
            counts[split] = (
                sum(1 for f in split_dir.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
                if split_dir.is_dir()
                else 0
            )
        rows.append((idx, folder.name, cleaned[idx] if idx < len(cleaned) else "?", counts))

    out = ROOT / "_pest_class_counts.tsv"
    lines = ["idx\tfolder\tclasses_line\tname\ttrain\tval\ttest\ttotal"]
    for idx, folder, name, counts in rows:
        total = sum(counts.values())
        lines.append(
            f"{idx}\t{folder}\t{idx + 1}\t{name}\t{counts['train']}\t{counts['val']}\t{counts['test']}\t{total}"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    grand = sum(sum(c.values()) for _, _, _, c in rows)
    print(f"classes={len(rows)}  images={grand}  -> {out.name}", flush=True)
    print(f"smallest class: {min(rows, key=lambda r: sum(r[3].values()))[2]} "
          f"({min(sum(r[3].values()) for r in rows)})", flush=True)
    print(f"largest class:  {max(rows, key=lambda r: sum(r[3].values()))[2]} "
          f"({max(sum(r[3].values()) for r in rows)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
