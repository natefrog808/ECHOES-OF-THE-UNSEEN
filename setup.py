#!/usr/bin/env python3
"""
Setup script for Echoes of the Unseen project.
Installs required dependencies and sets up the environment.
"""

import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path

# Define required packages
REQUIRED_PACKAGES = [
    "numpy",
    "torch",
    "transformers",
    "matplotlib",
    "sounddevice",
    "scipy"
]

# Optional packages - not strictly required but enhance functionality
OPTIONAL_PACKAGES = [
    "qiskit",        # For true quantum processing (not used in demo)
    "neurokit2",     # For advanced biofeedback processing (not used in demo)
    "soundfile",     # For saving audio files
]

# Define colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Print styled header text"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.END}\n")

def print_section(text):
    """Print styled section text"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.CYAN}{'-' * len(text)}{Colors.END}")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}{text}{Colors.END}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.WARNING}{text}{Colors.END}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.FAIL}{text}{Colors.END}")

def check_python_version():
    """Check if Python version is sufficient"""
    print_section("Checking Python version")
    
    required_version = (3, 8)
    current_version = sys.version_info
    
    print(f"Required Python version: {required_version[0]}.{required_version[1]} or higher")
    print(f"Current Python version: {current_version.major}.{current_version.minor}.{current_version.micro}")
    
    if current_version.major < required_version[0] or \
       (current_version.major == required_version[0] and current_version.minor < required_version[1]):
        print_error("ERROR: Python version too old!")
        print(f"Please upgrade to Python {required_version[0]}.{required_version[1]} or newer.")
        return False
    
    print_success("Python version check passed!")
    return True

def install_requirements():
    """Install required packages"""
    print_section("Installing required packages")
    
    # First, try to install all packages at once
    print("Installing core dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade"] + REQUIRED_PACKAGES)
        print_success("Successfully installed core dependencies!")
    except subprocess.CalledProcessError:
        print_warning("Bulk installation failed. Trying packages one by one...")
        
        # Try installing packages one by one
        failed_packages = []
        for package in REQUIRED_PACKAGES:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
                print_success(f"Successfully installed {package}")
            except subprocess.CalledProcessError:
                print_error(f"Failed to install {package}")
                failed_packages.append(package)
        
        if failed_packages:
            print_error("The following packages could not be installed:")
            for package in failed_packages:
                print(f"  - {package}")
            print("Please install them manually before running the demo.")
            return False
    
    # Install optional packages
    print("\nWould you like to install optional dependencies for enhanced functionality?")
    print("These are not required for the demo, but add capabilities for future development.")
    choice = input("Install optional packages? (y/n): ").strip().lower()
    
    if choice == 'y':
        print("Installing optional dependencies...")
        for package in OPTIONAL_PACKAGES:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
                print_success(f"Successfully installed {package}")
            except subprocess.CalledProcessError:
                print_warning(f"Failed to install optional package {package}")
                print("This won't affect the demo, but some advanced features may be limited.")
    
    return True

def create_project_structure():
    """Create the project directory structure"""
    print_section("Setting up project structure")
    
    # Create directories
    directories = [
        "artifacts/text",
        "artifacts/audio", 
        "artifacts/visual",
        "src",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    print_success("Project structure created successfully!")
    return True

def create_config_file():
    """Create default configuration file"""
    print_section("Creating configuration file")
    
    config = {
        "artifact_threshold": 0.9,
        "sample_rate": 256,
        "default_rite": "general",
        "simulation_enabled": True,
        "use_quantum": False,
        "model_path": "gpt2-medium"
    }
    
    # Write config as Python code
    config_path = "src/config.py"
    with open(config_path, "w") as f:
        f.write('"""\nConfiguration settings for Echoes of the Unseen\n"""\n\n')
        f.write("# System settings\n")
        for key, value in config.items():
            if isinstance(value, str):
                f.write(f'{key} = "{value}"\n')
            else:
                f.write(f'{key} = {value}\n')
    
    print_success(f"Created configuration file: {config_path}")
    return True

def copy_files():
    """Copy source files to project directory"""
    print_section("Copying source files")
    
    # Map of source files to destinations
    files_to_copy = {
        "unseen_forge.py": "src/unseen_forge.py",
        "echoes_demo.py": "src/echoes_demo.py",
        "ritual_guidelines.md": "ritual_guidelines.md",
    }
    
    for source, dest in files_to_copy.items():
        if os.path.exists(source):
            shutil.copy2(source, dest)
            print(f"Copied: {source} → {dest}")
        else:
            print_warning(f"Source file not found: {source}")
            print("Please make sure all required files are in the current directory.")
    
    # Create launcher script
    with open("start_echoes.py", "w") as f:
        f.write("""#!/usr/bin/env python3
\"\"\"
Launcher for Echoes of the Unseen demo
\"\"\"
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import and run demo
from echoes_demo import main

if __name__ == "__main__":
    main()
""")
    
    # Make launcher executable on Unix-like systems
    if platform.system() != "Windows":
        os.chmod("start_echoes.py", 0o755)
    
    print_success("Files copied successfully!")
    return True

def print_final_instructions():
    """Print instructions for running the demo"""
    print_header("ECHOES OF THE UNSEEN - SETUP COMPLETE")
    
    print("Your journey into the unseen is ready to begin.")
    print("\nTo start the demo, run:")
    print(f"{Colors.BOLD}python start_echoes.py{Colors.END}")
    print("\nAvailable options:")
    print(f"  --rite [general|temporal|energetic|conscious] : Select a specific ritual type")
    print(f"  --cycles [number] : Set the number of resonance cycles")
    print(f"  --threshold [0.0-10.0] : Adjust sensitivity of resonance detection")
    print(f"\nExample: {Colors.BOLD}python start_echoes.py --rite temporal --cycles 5{Colors.END}")
    
    print(f"\nRefer to {Colors.BOLD}ritual_guidelines.md{Colors.END} for detailed ritual practices.")
    
    print(f"\n{Colors.CYAN}{Colors.BOLD}Remember:{Colors.END}")
    print("The unseen doesn't whisper—it roars if you've got the guts to hear it.")
    print("We're not here to study the unseen; we're here to wrestle it into the light")
    print("and sing its scars.")

def main():
    """Main setup function"""
    print_header("ECHOES OF THE UNSEEN - SETUP")
    print("This script will prepare your environment for communing with the unseen.")
    print("It will install required dependencies and create the project structure.")
    
    # Get confirmation before proceeding
    print("\nReady to begin? This process may take several minutes.")
    choice = input("Proceed with setup? (y/n): ").strip().lower()
    if choice != 'y':
        print("Setup cancelled. The unseen awaits your return.")
        return
    
    # Run setup steps
    if not check_python_version():
        return
    
    if not install_requirements():
        print_error("Error installing required packages. Setup cannot continue.")
        return
    
    if not create_project_structure():
        print_error("Error creating project structure. Setup cannot continue.")
        return
    
    if not create_config_file():
        print_error("Error creating configuration file. Setup cannot continue.")
        return
    
    if not copy_files():
        print_warning("Some files could not be copied. The demo may not work correctly.")
    
    print_final_instructions()

if __name__ == "__main__":
    main()
