import subprocess
import zipfile
import shutil
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

def download_torrent(torrent: Path, download_dir: Path):  
    # --seed-time=0 drops the torrent immediately after finishing the download
    cmd = ["aria2c", "--select-file=119", "--seed-time=0", f"--dir={download_dir}", torrent.as_posix()]
    
    subprocess.run(cmd, check=True)

def extract_zip(zip_path: Path, extract_to: Path):
    with zipfile.ZipFile(zip_path, 'r') as zip:
        zip.extractall(zip_path.parent)

    files_paths = list((zip_path.parent / "jeeproject@yahoo.com tranche 1").iterdir())
    for file_path in files_paths:
        shutil.copyfile(file_path, extract_to / file_path.name)

def cleanup(target_dir: Path):
    shutil.rmtree(target_dir)

if __name__ == "__main__":
    TORRENT = Path("data/Epstein files 2026-02-11.torrent")
    DOWNLOAD_FOLDER = Path("data/download")
    DOWNLOADED_ZIP = DOWNLOAD_FOLDER / "Epstein files 2026-02-11/Jeffrey Epstein/Emails/jeeproject@yahoo.com/jeeproject@yahoo.com tranche 1.zip"
    DESTINATION = Path("data/raw/")
    DESTINATION.mkdir(exist_ok=True)
    console = Console()
    
    console.print(Panel("[magenta]DOWNLOADING DATASET"))
    
    console.print(Panel.fit("[cyan]DOWNLOADING TORRENT..."))
    download_torrent(TORRENT, DOWNLOAD_FOLDER)
    console.print("[green]Download completed")
    
    console.print(Panel.fit("[cyan]EXTRACTING ZIP..."))
    extract_zip(DOWNLOADED_ZIP, DESTINATION)
    console.print("[green]Done")
    
    console.print(Panel.fit("[cyan]CLEANING UP..."))
    cleanup(DOWNLOAD_FOLDER)
    console.print("[green]All  Done")