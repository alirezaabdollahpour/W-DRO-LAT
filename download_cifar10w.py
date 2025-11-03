#!/usr/bin/env python3

"""Utility script to download and unpack the CIFAR-10-W testbed dataset.

The script downloads a curated collection of CIFAR-10 distribution shifts that
were released alongside the CIFAR-10-W benchmark. Each archive is hosted on
Google Drive and fetched with ``gdown`` by file id. After the download completes
the archives are extracted into the target directory and (optionally) deleted.

Example:
    python download_cifar10w.py --out /path/to/cifar10w

The default behaviour mirrors running the shell commands supplied in the
benchmark release notes while adding a few quality-of-life features:
    * automatic installation/upgrade of gdown
    * resumable downloads (``gdown`` handles this)
    * skip logic when archives already exist
    * optional re-use of the downloaded ``.zip`` files
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
import zipfile


@dataclass(frozen=True)
class DatasetArchive:
    """Static metadata describing each CIFAR-10-W archive."""

    filename: str
    file_id: str
    size_gb: Optional[float] = None
    notes: Optional[str] = None

    @property
    def stem(self) -> str:
        return self.filename.rsplit(".", maxsplit=1)[0]


ARCHIVES: List[DatasetArchive] = [
    DatasetArchive("bing_cartoon_original.zip", "18RKtnxfKGiFTXvDsT4nd9ve04bUt2J3H", 1.12),
    DatasetArchive("data_360_cartoon_original.zip", "1yz56SW9NJzqcon-2Jj2Slnb9ANPHbC28", 0.55),
    DatasetArchive("data_360_original.zip", "1YSGQtxcmpnSiHsveAyR5TR2l_5vCzxKl", 1.26),
    DatasetArchive("data_baidu_cartoon_original.zip", "1b9E7W_gu1845DADALJmB9UulofsoCDoW", 1.48),
    DatasetArchive("data_baidu_original.zip", "1aT1zlgEQxuL2FjlZTIZpXDyVgWhUy4Fi", 5.08),
    DatasetArchive("data_bing_original.zip", "1fIVQa0Ma04B2n-_RdwOqQkt2rV7Zdo9U", 3.43),
    DatasetArchive("data_flickr_original.zip", "1p3zzK5M4elIXAbcLKNXXSIu3Uiu9SxL5", 2.54),
    DatasetArchive("data_google_original.zip", "1ITAS0ESiLfP9oRkT3OXoYByI5iDbB9fd", 7.07),
    DatasetArchive("data_pexel_original.zip", "113T3SMgIBfZJIw7XxZaXoOfIIUyPU4WL", 1.08),
    DatasetArchive("data_sougou_original.zip", "1WravuqPV1UrSEaoP6bAlu7ZvxwbNHMpX", 1.04),
    DatasetArchive("diffusion_cartoon_original.zip", "1zH8zW1ri6bL4Zgb0lsdZM-EeM_aTZYHO", 4.62),
    DatasetArchive("diffusion_hard_original.zip", "1UMAFveY2gTJcebEn56OUDzhoCtc6uUi7", 3.94),
    DatasetArchive("diffusion_original.zip", "1cQyStZe0u-t9sjjUitqioP5OO9Qfalga", 2.11),
]


class CIFAR10WDownloader:
    """Download helper encapsulating gdown usage and archive extraction."""

    def __init__(
        self,
        output_dir: Path,
        keep_zip: bool = False,
        skip_existing: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.keep_zip = keep_zip
        self.skip_existing = skip_existing
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_gdown_available()

    def _ensure_gdown_available(self) -> None:
        """Install/upgrade gdown so large downloads can resume if interrupted."""
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "gdown"],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            print(f"Warning: Failed to (re)install gdown: {exc}")

    def download_archives(self, archives: Iterable[DatasetArchive]) -> None:
        """Download every archive with gdown if missing or incomplete."""
        for archive in archives:
            target_zip = self.output_dir / archive.filename
            extracted_dir = self.output_dir / archive.stem

            if target_zip.exists() and target_zip.stat().st_size > 1024 * 1024:
                if self.skip_existing:
                    print(f"Skipping existing archive {archive.filename}")
                else:
                    print(f"Re-downloading {archive.filename} (file exists)")
                    self._download_file(archive, target_zip)
            else:
                self._download_file(archive, target_zip)

            if extracted_dir.exists() and any(extracted_dir.iterdir()):
                if self.skip_existing:
                    print(f"Found extracted data for {archive.stem}; skipping unpack.")
                    continue
                print(f"Re-extracting {archive.stem}")

            self._extract_archive(target_zip, extracted_dir)

            if not self.keep_zip and target_zip.exists():
                target_zip.unlink()
                print(f"Removed archive {archive.filename}")

    def _download_file(self, archive: DatasetArchive, output_path: Path) -> None:
        print(f"Downloading {archive.filename} ({archive.file_id})")
        cmd = [
            sys.executable,
            "-m",
            "gdown",
            "--id",
            archive.file_id,
            "-O",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Failed to download {archive.filename}: {exc}")
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            raise

    def _extract_archive(self, zip_path: Path, target_dir: Path) -> None:
        print(f"Extracting {zip_path.name}")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(target_dir)
        except zipfile.BadZipFile as exc:
            print(f"Extraction failed for {zip_path}: {exc}")
            raise


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and unpack the CIFAR-10-W dataset collection.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.cwd() / "cifar10w",
        help="Destination directory for the dataset (default: ./cifar10w).",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded .zip archives instead of deleting them after extraction.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Force re-download and re-extraction even if files already exist.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the available archives and exit.",
    )
    parser.add_argument(
        "--archives",
        nargs="+",
        choices=[archive.filename for archive in ARCHIVES],
        help="Optional subset of archives to download (provide space-separated filenames).",
    )
    return parser.parse_args(argv)


def list_archives() -> None:
    print("CIFAR-10-W archives available for download:")
    for archive in ARCHIVES:
        size = f"{archive.size_gb:.2f} GB" if archive.size_gb else "unknown size"
        print(f"  - {archive.filename:30s} {size}")


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if args.list:
        list_archives()
        return

    selected_archives = (
        [archive for archive in ARCHIVES if archive.filename in args.archives]
        if args.archives
        else ARCHIVES
    )

    downloader = CIFAR10WDownloader(
        output_dir=args.out,
        keep_zip=args.keep_zip,
        skip_existing=args.skip_existing,
    )
    downloader.download_archives(selected_archives)


if __name__ == "__main__":
    main()
