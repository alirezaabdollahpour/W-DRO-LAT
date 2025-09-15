#!/usr/bin/env python3
"""
PyTorch Checkpoint Downloader for Google Drive
Optimized for downloading .pt checkpoint files from Google Drive folders.
Supports multiple download methods with fallback strategies.
"""

import os
import sys
import subprocess
import requests
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm for progress tracking...")
    subprocess.run([sys.executable, "-m", "pip", "install", "tqdm"], check=True)
    from tqdm import tqdm


@dataclass
class CheckpointFile:
    """Data class for checkpoint file information."""
    name: str
    file_id: Optional[str] = None
    size: Optional[int] = None
    downloaded: bool = False


class PyTorchCheckpointDownloader:
    """
    A robust downloader for PyTorch checkpoint files from Google Drive.
    
    Supports multiple download methods:
    1. gdown with folder download
    2. gdown with individual file IDs
    3. curl with cookie handling
    4. wget with authentication
    """
    
    def __init__(self, download_path: str, timeout: int = 600):
        """
        Initialize the checkpoint downloader.
        
        Args:
            download_path: Local directory to save checkpoint files
            timeout: Timeout for download operations in seconds
        """
        self.download_path = Path(download_path)
        self.download_path.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.session = requests.Session()
        
        # Expected checkpoint files (can be customized)
        self.expected_files = [
            CheckpointFile("vitb16_finetune.pt"),
            CheckpointFile("vitb16_pretrain.pt"),
            CheckpointFile("vitb32_finetune.pt"),
            CheckpointFile("vitb32_pretrain.pt"),
            CheckpointFile("vit14_finetune.pt"),
            CheckpointFile("vit14_pretrain.pt")
        ]
        
        self._ensure_dependencies()
    
    def _ensure_dependencies(self) -> None:
        """Ensure required dependencies are installed."""
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-U", "gdown"], 
                         check=True, capture_output=True)
            print("✓ Dependencies verified")
        except subprocess.CalledProcessError as e:
            print(f"⚠ Warning: Could not update gdown: {e}")
    
    def extract_folder_id(self, url: str) -> str:
        """Extract folder ID from Google Drive URL."""
        if '/folders/' in url:
            return url.split('/folders/')[1].split('?')[0].split('/')[0]
        elif '/file/d/' in url:
            return url.split('/file/d/')[1].split('/')[0]
        else:
            # Try to extract ID from the end of URL
            return url.split('/')[-1].split('?')[0]
    
    def check_existing_files(self) -> List[CheckpointFile]:
        """Check which files already exist and their status."""
        existing_files = []
        
        for checkpoint in self.expected_files:
            file_path = self.download_path / checkpoint.name
            if file_path.exists() and file_path.stat().st_size > 1000:  # Larger than 1KB
                checkpoint.downloaded = True
                checkpoint.size = file_path.stat().st_size
                existing_files.append(checkpoint)
                print(f"✓ Found existing: {checkpoint.name} ({checkpoint.size / 1024 / 1024:.1f} MB)")
        
        return existing_files
    
    def download_with_gdown_folder(self, folder_id: str) -> bool:
        """
        Method 1: Download entire folder using gdown.
        
        Args:
            folder_id: Google Drive folder ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        print("🔄 Attempting folder download with gdown...")
        
        try:
            cmd = [
                "gdown", 
                "--folder", 
                "--id", folder_id,
                "-O", str(self.download_path),
                "--remaining-ok"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            
            if result.returncode == 0:
                # Check if .pt files were downloaded
                pt_files = list(self.download_path.glob("*.pt"))
                if pt_files:
                    print(f"✓ Successfully downloaded {len(pt_files)} files using folder method")
                    for file in pt_files:
                        print(f"  - {file.name} ({file.stat().st_size / 1024 / 1024:.1f} MB)")
                    return True
            
            print(f"⚠ Folder download failed: {result.stderr}")
            return False
            
        except subprocess.TimeoutExpired:
            print("⚠ Folder download timed out")
            return False
        except Exception as e:
            print(f"⚠ Folder download error: {e}")
            return False
    
    def download_with_gdown_individual(self, folder_id: str) -> Dict[str, bool]:
        """
        Method 2: Download individual files using gdown with fuzzy matching.
        
        Args:
            folder_id: Google Drive folder ID
            
        Returns:
            Dict[str, bool]: Mapping of filename to success status
        """
        print("🔄 Attempting individual file downloads with gdown...")
        
        results = {}
        
        for checkpoint in self.expected_files:
            if checkpoint.downloaded:
                results[checkpoint.name] = True
                continue
                
            print(f"  Downloading {checkpoint.name}...")
            output_path = self.download_path / checkpoint.name
            
            try:
                # Try fuzzy download
                drive_url = f"https://drive.google.com/drive/folders/{folder_id}"
                cmd = [
                    "gdown", 
                    "--fuzzy", 
                    drive_url,
                    "-O", str(output_path)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
                
                if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
                    checkpoint.downloaded = True
                    checkpoint.size = output_path.stat().st_size
                    results[checkpoint.name] = True
                    print(f"    ✓ Downloaded {checkpoint.name} ({checkpoint.size / 1024 / 1024:.1f} MB)")
                else:
                    results[checkpoint.name] = False
                    print(f"    ✗ Failed to download {checkpoint.name}")
                    
            except Exception as e:
                results[checkpoint.name] = False
                print(f"    ✗ Error downloading {checkpoint.name}: {e}")
        
        return results
    
    def download_with_curl(self, folder_id: str) -> Dict[str, bool]:
        """
        Method 3: Download using curl with cookie handling.
        
        Args:
            folder_id: Google Drive folder ID
            
        Returns:
            Dict[str, bool]: Mapping of filename to success status
        """
        print("🔄 Attempting downloads with curl...")
        
        results = {}
        cookie_file = self.download_path / "cookies.txt"
        
        for checkpoint in self.expected_files:
            if checkpoint.downloaded:
                results[checkpoint.name] = True
                continue
                
            print(f"  Downloading {checkpoint.name} with curl...")
            output_path = self.download_path / checkpoint.name
            
            try:
                # Create curl command with proper cookie handling
                curl_cmd = [
                    "curl", "-L", "-c", str(cookie_file),
                    "-o", str(output_path),
                    f"https://drive.google.com/uc?export=download&id={folder_id}&confirm=t"
                ]
                
                result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=self.timeout)
                
                if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
                    checkpoint.downloaded = True
                    checkpoint.size = output_path.stat().st_size
                    results[checkpoint.name] = True
                    print(f"    ✓ Downloaded {checkpoint.name} ({checkpoint.size / 1024 / 1024:.1f} MB)")
                else:
                    results[checkpoint.name] = False
                    print(f"    ✗ Failed to download {checkpoint.name}")
                    
            except Exception as e:
                results[checkpoint.name] = False
                print(f"    ✗ Error downloading {checkpoint.name}: {e}")
        
        # Clean up cookie file
        if cookie_file.exists():
            cookie_file.unlink()
        
        return results
    
    def download_with_wget(self, folder_id: str) -> Dict[str, bool]:
        """
        Method 4: Download using wget with authentication.
        
        Args:
            folder_id: Google Drive folder ID
            
        Returns:
            Dict[str, bool]: Mapping of filename to success status
        """
        print("🔄 Attempting downloads with wget...")
        
        results = {}
        cookie_file = "/tmp/gdrive_cookies.txt"
        
        for checkpoint in self.expected_files:
            if checkpoint.downloaded:
                results[checkpoint.name] = True
                continue
                
            print(f"  Downloading {checkpoint.name} with wget...")
            output_path = self.download_path / checkpoint.name
            
            try:
                # Complex wget command for Google Drive
                wget_cmd = f'''wget --load-cookies {cookie_file} "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies {cookie_file} --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id={folder_id}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id={folder_id}" -O {output_path}'''
                
                result = subprocess.run(wget_cmd, shell=True, capture_output=True, text=True, timeout=self.timeout)
                
                if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
                    checkpoint.downloaded = True
                    checkpoint.size = output_path.stat().st_size
                    results[checkpoint.name] = True
                    print(f"    ✓ Downloaded {checkpoint.name} ({checkpoint.size / 1024 / 1024:.1f} MB)")
                else:
                    results[checkpoint.name] = False
                    print(f"    ✗ Failed to download {checkpoint.name}")
                    
            except Exception as e:
                results[checkpoint.name] = False
                print(f"    ✗ Error downloading {checkpoint.name}: {e}")
        
        # Clean up cookie file
        if os.path.exists(cookie_file):
            os.remove(cookie_file)
        
        return results
    
    def download_checkpoints(self, drive_url: str) -> bool:
        """
        Main method to download PyTorch checkpoints with multiple fallback strategies.
        
        Args:
            drive_url: Google Drive folder or file URL
            
        Returns:
            bool: True if at least some files were downloaded successfully
        """
        folder_id = self.extract_folder_id(drive_url)
        print(f"📥 Starting checkpoint download from folder ID: {folder_id}")
        print(f"📁 Download path: {self.download_path.absolute()}")
        
        # Check existing files
        existing_files = self.check_existing_files()
        if len(existing_files) == len(self.expected_files):
            print("✅ All checkpoint files already exist!")
            return True
        
        # Try download methods in order of preference
        download_methods = [
            ("Folder Download", self.download_with_gdown_folder),
            ("Individual gdown", lambda fid: self.download_with_gdown_individual(fid)),
            ("Curl Download", lambda fid: self.download_with_curl(fid)),
            ("Wget Download", lambda fid: self.download_with_wget(fid))
        ]
        
        for method_name, method_func in download_methods:
            print(f"\n{'='*60}")
            print(f"Trying: {method_name}")
            print(f"{'='*60}")
            
            try:
                if method_name == "Folder Download":
                    success = method_func(folder_id)
                    if success:
                        break
                else:
                    results = method_func(folder_id)
                    if any(results.values()):
                        print(f"✓ {method_name} had partial success")
                        # Check if we have all files now
                        self.check_existing_files()
                        if all(cp.downloaded for cp in self.expected_files):
                            break
            except Exception as e:
                print(f"✗ {method_name} failed with error: {e}")
                continue
        
        # Final status check
        return self._print_final_status(folder_id)
    
    def _print_final_status(self, folder_id: str) -> bool:
        """Print final download status and instructions if needed."""
        downloaded_files = [cp for cp in self.expected_files if cp.downloaded]
        missing_files = [cp for cp in self.expected_files if not cp.downloaded]
        
        print(f"\n{'='*80}")
        print("DOWNLOAD SUMMARY")
        print(f"{'='*80}")
        
        if downloaded_files:
            print(f"✅ Successfully downloaded {len(downloaded_files)} files:")
            total_size = 0
            for cp in downloaded_files:
                size_mb = cp.size / 1024 / 1024 if cp.size else 0
                total_size += size_mb
                print(f"  ✓ {cp.name} ({size_mb:.1f} MB)")
            print(f"📊 Total size: {total_size:.1f} MB")
        
        if missing_files:
            print(f"\n❌ Failed to download {len(missing_files)} files:")
            for cp in missing_files:
                print(f"  ✗ {cp.name}")
            
            self._print_manual_instructions(folder_id)
            return len(downloaded_files) > 0
        else:
            print("\n🎉 All checkpoint files downloaded successfully!")
            return True
    
    def _print_manual_instructions(self, folder_id: str) -> None:
        """Print manual download instructions."""
        print(f"\n{'='*80}")
        print("MANUAL DOWNLOAD INSTRUCTIONS")
        print(f"{'='*80}")
        print("Some files couldn't be downloaded automatically. Manual options:")
        print(f"\n1. Browser download:")
        print(f"   Visit: https://drive.google.com/drive/folders/{folder_id}")
        print(f"   Download files manually and transfer to: {self.download_path.absolute()}")
        
        print(f"\n2. Direct gdown commands:")
        print(f"   gdown --folder --id {folder_id} -O {self.download_path}")
        
        print(f"\n3. Individual file download (if you have file IDs):")
        print(f"   gdown --id FILE_ID -O {self.download_path}/filename.pt")
        
        print(f"\n4. Using scp from local machine:")
        print(f"   scp /path/to/downloaded/file.pt user@server:{self.download_path}/")
        print(f"{'='*80}")


def main():
    """Main function to run the checkpoint downloader."""
    
    # Configuration
    DRIVE_URL = "https://drive.google.com/drive/folders/1dwfb2Iqw6BTMi57SeG44aDebuu-TBzo5?usp=drive_link"
    DOWNLOAD_PATH = "/mnt/lts4/scratch/students/aabdolla/LAT/ResNet_checkpoints"
    
    # Option 1: Auto-discover files in the folder (recommended)
    downloader = PyTorchCheckpointDownloader(
        download_path=DOWNLOAD_PATH,
        timeout=600  # 10 minutes timeout
    )
    
    # Option 2: Specify exact files if you know them (uncomment and modify as needed)
    # specific_files = ["checkpoint_epoch_100.pt", "best_model.pt", "final_weights.pt"]
    # downloader = PyTorchCheckpointDownloader(
    #     download_path=DOWNLOAD_PATH,
    #     timeout=600,
    #     expected_files=specific_files
    # )
    
    # Start download
    success = downloader.download_checkpoints(DRIVE_URL)
    
    if success:
        print("\n🎉 Checkpoint download process completed!")
    else:
        print("\n⚠ Download process completed with some failures. Check manual instructions above.")


if __name__ == "__main__":
    main()