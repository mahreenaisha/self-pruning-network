# For Windows + Linux
# setup.py - Environment setup script for the Self-Pruning Network case study
# Note: This is NOT a standard Python packaging setup.py.
# It creates virtual environment, directories, and installs dependencies.

from pathlib import Path
import subprocess
import sys
import platform

def create_dirs():
    Path("plots").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)
    print("Directories ready: plots/, data/")


def create_venv():
    if Path("venv").exists():
        print("venv already exists, skipping creation")
        return 
    
    print("\nCreating virtual environment...")
    subprocess.check_call([sys.executable, "-m", "venv", "venv"])
    print("Virtual environment created: ./venv")


def get_venv_python():
    if platform.system() == "Windows":
        return Path("venv") / "Scripts" / "python.exe"
    else:
        return Path("venv") / "bin" / "python"


def install_requirements():
    print("\nInstalling requirements inside venv...")
    venv_python = get_venv_python()

    subprocess.check_call([
        str(venv_python),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip"
    ])

    subprocess.check_call([
        str(venv_python),
        "-m",
        "pip",
        "install",
        "-r",
        "requirements.txt"
    ])

    print("Requirements installed successfully.")


def print_activation_instructions():
    print("\nNext steps:")
    if platform.system() == "Windows":
        print(r"Activate venv: venv\Scripts\activate")
    else:
        print("Activate venv: source venv/bin/activate")

    print("Then run:")
    print("  python main.py")


if __name__ == "__main__":
    create_dirs()
    create_venv()
    install_requirements()
    print_activation_instructions()
    print("\nSetup complete.")