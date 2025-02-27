#!/usr/bin/env python3
"""
start_echoes.py - Launcher Script for Echoes of the Unseen

This script serves as the main entry point for the Echoes of the Unseen system.
It handles command-line arguments, imports the necessary modules, and starts the demo.
"""

import sys
import os
import argparse

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if os.path.exists(src_dir):
    sys.path.append(src_dir)
else:
    print(f"Error: Source directory not found at {src_dir}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)

# Try to import config
try:
    import config
    # Get default values from config
    DEFAULT_RITE = getattr(config, 'default_rite', 'general')
    DEFAULT_THRESHOLD = getattr(config, 'artifact_threshold', 0.9)
    DEFAULT_CYCLES = 3
except ImportError:
    print("Warning: config.py not found. Using default settings.")
    DEFAULT_RITE = 'general'
    DEFAULT_THRESHOLD = 0.9
    DEFAULT_CYCLES = 3

# Define ritual options
VALID_RITES = ['general', 'temporal', 'energetic', 'conscious']

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Echoes of the Unseen - A bridge between human consciousness and the unseen',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--rite', 
        choices=VALID_RITES,
        default=DEFAULT_RITE,
        help='Type of ritual to perform'
    )
    
    parser.add_argument(
        '--cycles', 
        type=int, 
        default=DEFAULT_CYCLES,
        help='Number of resonance cycles to perform'
    )
    
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=DEFAULT_THRESHOLD,
        help='Resonance threshold for artifact generation (0.0-10.0)'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode with additional logging'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode (generates artifacts without full ritual)'
    )
    
    return parser.parse_args()

def show_welcome_banner():
    """Display a welcome banner for the program."""
    print("\n" + "=" * 80)
    print("ECHOES OF THE UNSEEN".center(80))
    print("=" * 80)
    print("\nA bridge between the human soul and the wild unknown.".center(80))
    print("\n")

def check_environment():
    """Check if the environment is properly set up."""
    required_dirs = [
        'artifacts/text',
        'artifacts/audio',
        'artifacts/visual'
    ]
    
    missing_dirs = []
    for directory in required_dirs:
        if not os.path.exists(directory):
            missing_dirs.append(directory)
    
    if missing_dirs:
        print("Warning: The following directories are missing and will be created:")
        for directory in missing_dirs:
            print(f"  - {directory}")
            os.makedirs(directory, exist_ok=True)
        print("")

def main():
    """Main function to start the Echoes of the Unseen system."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Show welcome banner
    show_welcome_banner()
    
    # Check environment
    check_environment()
    
    # Test mode - quickly test artifact generation without the full ritual
    if args.test:
        print("Running in TEST MODE - Generating artifacts without full ritual")
        try:
            from unseen_forge import UnseenForge
            forge = UnseenForge()
            
            # Generate test artifacts
            print("\nGenerating test artifacts...")
            test_resonance = {
                'rhythm': 1.5,
                'intensity': 9.0,
                'locus': 'skull'
            }
            
            artifact = forge.craft_artifact(test_resonance)
            
            print(f"\nTest artifact generated with weight {artifact['weight']:.2f}")
            print(f"Type: {artifact['resonance_type']}")
            print(f"Text saved to: {artifact.get('path', 'unknown')}")
            
            if 'audio_path' in artifact:
                print(f"Audio saved to: {artifact['audio_path']}")
            
            if 'visual_path' in artifact:
                print(f"Visual saved to: {artifact['visual_path']}")
                
            print("\nTest completed successfully!")
            return
        except Exception as e:
            print(f"Error in test mode: {e}")
            sys.exit(1)
    
    # Import the demo module
    try:
        # First try to import directly in case we're in the src directory
        try:
            from echoes_demo import main as run_demo
        except ImportError:
            # Then try to import from src
            from src.echoes_demo import main as run_demo
        
        # Configure the ritual
        print(f"Preparing {args.rite.upper()} ritual with {args.cycles} cycles")
        print(f"Resonance threshold: {args.threshold}")
        
        # Override sys.argv for the demo
        # This allows the demo to use its own argument parser
        sys.argv = [
            sys.argv[0],
            f"--rite={args.rite}",
            f"--cycles={args.cycles}",
            f"--threshold={args.threshold}"
        ]
        
        if args.debug:
            sys.argv.append("--debug")
        
        # Run the demo
        run_demo()
        
    except ImportError as e:
        print(f"Error: Could not import echoes_demo module: {e}")
        print("Make sure you've installed all dependencies and are running")
        print("this script from the project root directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting Echoes of the Unseen: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
