import argparse
import os
import tarfile
import zipfile
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    _HAS_KAGGLE = True
except Exception:
    _HAS_KAGGLE = False


def run_subprocess(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    cp = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if cp.returncode != 0:
        print("Subprocess failed:")
        print(cp.stdout)
        print(cp.stderr)
        raise RuntimeError("Subprocess failed")
    return cp.stdout


def download_all_from_kaggle(competition, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    # Try Python KaggleApi if available
    if _HAS_KAGGLE:
        try:
            api = KaggleApi()
            api.authenticate()
            print("Downloading files with Kaggle API (may take a while)...")
            api.competition_download_files(competition, path=str(dest), quiet=False)
            print("Kaggle API download finished.")
            return
        except Exception as e:
            print("KaggleApi failed:", e)
            print("Falling back to kaggle CLI (subprocess).")

    # Fallback to kaggle CLI
    run_subprocess([
        "kaggle", "competitions", "download",
        "-c", competition, "-p", str(dest), "--force"
    ])


def extract_archives(dest, remove_archives=False):
    dest = Path(dest)
    archives = list(dest.glob("*.tar")) + list(dest.glob("*.tar.gz")) + list(dest.glob("*.tgz")) + list(dest.glob("*.zip"))
    if not archives:
        print("No archives found in", dest)
        return

    for arc in archives:
        print(f"Extracting {arc} ...")
        if arc.suffix == ".zip":
            with zipfile.ZipFile(arc, "r") as zf:
                zf.extractall(path=dest)
        else:
            with tarfile.open(arc, "r:*") as tf:
                members = tf.getmembers()
                for m in tqdm(members, desc=f"Extracting {arc.name}", unit="file"):
                    try:
                        tf.extract(m, path=dest)
                    except Exception:
                        pass
        if remove_archives:
            try:
                arc.unlink()
            except Exception:
                pass


def parse_split_file(file_path):
    """
    Parse ILSVRC/ImageSets/CLS-LOC/*.txt split files.
    Lines look like:
        n01440764/n01440764_10026 1
    Return list of (image_path, class_label).
    """
    mapping = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                rel_path, label = parts[0], parts[1]
                mapping.append((rel_path, label))
    return mapping


def prepare_split(split_name, images_root, split_file, out_root):
    print(f"Preparing {split_name} split...")
    out_dir = out_root / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = parse_split_file(split_file)

    moved = 0
    for rel_path, label in tqdm(mapping, desc=f"{split_name} images"):
        img_src = images_root / "ILSVRC" / "Data" / "CLS-LOC" / split_name / (rel_path + ".JPEG")
        if not img_src.exists():
            # Some files might be missing or named differently
            continue
        target_dir = out_dir / label
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_src, target_dir / img_src.name)
        moved += 1
    print(f"{split_name}: moved {moved} images into {out_dir}")


def summarize_structure(dest):
    dest = Path(dest)

    def count_images(p):
        return sum(1 for _ in p.rglob("*.JPEG"))

    print("Summary of prepared dataset (top-level):")
    for split in ["train", "val", "test"]:
        path = dest / split
        if path.exists():
            classes = [d for d in path.iterdir() if d.is_dir()]
            print(f"  {split}: classes={len(classes)}, images={count_images(path)}, path={path}")
        else:
            print(f"  {split}: NOT FOUND")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare ImageNet CLS-LOC for classification (ImageFolder format).")
    parser.add_argument("--competition", type=str, default="imagenet-object-localization-challenge",
                        help="Kaggle competition slug")
    parser.add_argument("--dest", type=str, default="~/data/imagenet",
                        help="Destination folder for download/extraction/preparation")
    parser.add_argument("--no-download", action="store_true", help="Skip download (assume files already present)")
    parser.add_argument("--keep-archives", action="store_true", help="Do not delete archives after extraction")
    args = parser.parse_args()

    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if not args.no_download:
        print("Starting download")
        download_all_from_kaggle(args.competition, dest)

    print("Extracting archives")
    extract_archives(dest, remove_archives=not args.keep_archives)

    # Build splits
    set_root = dest / "ILSVRC" / "ImageSets" / "CLS-LOC"
    images_root = dest

    if set_root.exists():
        for split_name, split_file in [
            ("train", set_root / "train_cls.txt"),
            ("val", set_root / "val.txt"),
            ("test", set_root / "test.txt"),
        ]:
            if split_file.exists():
                prepare_split(split_name, images_root, split_file, dest)
            else:
                print(f"Split file not found: {split_file}")
    else:
        print("Could not find ImageSets/CLS-LOC directory. Please check extraction.")

    summarize_structure(dest)


if __name__ == "__main__":
    main()
