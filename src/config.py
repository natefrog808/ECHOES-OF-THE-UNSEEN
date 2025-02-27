"""
Configuration settings for Echoes of the Unseen

This file contains all configurable parameters for the system.
Edit these values to customize your experience.
"""

# System settings
#-------------------------------------------------------------------------------

# Resonance threshold for triggering artifact generation (0.0-10.0)
# Higher values require stronger resonance signals
artifact_threshold = 0.9

# Sample rate for EEG data processing (Hz)
sample_rate = 256

# Default ritual type ('general', 'temporal', 'energetic', 'conscious')
default_rite = "general"

# Enable simulation mode (True) or use real hardware (False)
simulation_enabled = True

# Use actual quantum computing via IBM Quantum (requires API token)
use_quantum = False
quantum_token = ""  # Your IBM Quantum token if use_quantum is True

# Language model settings
#-------------------------------------------------------------------------------

# Path to language model for artifact generation
# Options:
#   - "distilgpt2" (smaller, faster, less quality)
#   - "gpt2" (balanced)
#   - "gpt2-medium" (higher quality, slower)
#   - "/path/to/your/fine-tuned/model" (custom model)
model_path = "distilgpt2"

# Generation parameters
max_length = 100        # Maximum length of generated text
temperature_base = 0.7  # Base temperature for generation (higher = more random)
top_k = 40              # Number of highest probability tokens to consider

# Audio settings
#-------------------------------------------------------------------------------

# Enable audio generation for significant artifacts
audio_enabled = True

# Minimum resonance weight to trigger audio generation
audio_threshold = 1.2

# Sample rate for audio generation (Hz)
audio_sample_rate = 44100

# Visual settings
#-------------------------------------------------------------------------------

# Enable visual generation for powerful artifacts
visual_enabled = True

# Minimum resonance weight to trigger visual generation
visual_threshold = 1.8

# Visual resolution (DPI) for saved images
visual_dpi = 150

# Hardware settings (when simulation_enabled = False)
#-------------------------------------------------------------------------------

# Hardware type to use ('openbci', 'muse', 'emotiv', or 'custom')
hardware_type = "openbci"

# Connection parameters
serial_port = ""        # COM port for serial devices
ip_address = ""         # IP address for network devices
bluetooth_address = ""  # Bluetooth address for wireless devices

# Debugging settings
#-------------------------------------------------------------------------------

# Enable verbose logging
debug_mode = False

# Log file path (empty for no logging)
log_file = "logs/echoes.log"

# Automatically save all session data
save_sessions = True
