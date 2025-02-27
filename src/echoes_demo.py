#!/usr/bin/env python3
"""
Echoes of the Unseen - Demo Script
A prototype of the Echoes system that simulates biofeedback and quantum processing
to generate resonant artifacts.

This demo implements a simplified version of the full system, using classical algorithms
to simulate quantum processing and generated biofeedback data instead of real sensors.
It demonstrates the core ritual experience of resonating with the unseen and generating artifacts.
"""

import numpy as np
import time
import threading
import random
import json
import os
from datetime import datetime
from queue import Queue
import argparse
import signal
import sys

# Import the UnseenForge component
# Note: In a real implementation, this would be imported from a module
# For the demo, we assume UnseenForge is in a file called unseen_forge.py
try:
    from unseen_forge import UnseenForge
except ImportError:
    print("Error: Cannot import UnseenForge. Make sure unseen_forge.py is in the same directory.")
    sys.exit(1)

# Simplified QuantumEchoLayer for demo - uses classical probability instead of Qiskit
class SimpleQuantumEcho:
    def __init__(self, dimensions=6):
        """Initialize a simplified quantum-inspired processing layer
        
        Args:
            dimensions: Number of dimensions to use in processing
        """
        self.dimensions = dimensions
        print(f"Initializing SimpleQuantumEcho with {dimensions} dimensions")
        
    def weave_echo(self, input_state):
        """Process input through a quantum-inspired algorithm
        
        Args:
            input_state: Array of input values
        
        Returns:
            Array of processed values
        """
        # Normalize input
        if len(input_state) < self.dimensions:
            # Pad with zeros if needed
            input_state = np.pad(input_state, (0, self.dimensions - len(input_state)))
        else:
            # Truncate if too long
            input_state = input_state[:self.dimensions]
            
        # Normalize to unit vector (similar to quantum state)
        norm = np.linalg.norm(input_state)
        if norm > 0:
            normalized = input_state / norm
        else:
            normalized = np.ones(self.dimensions) / np.sqrt(self.dimensions)
        
        # Apply a "superposition-like" transformation
        # This creates a probability distribution based on input
        probabilities = normalized ** 2
        
        # Apply an "interference-like" effect
        # This introduces correlations between dimensions
        interference = np.zeros_like(probabilities)
        for i in range(self.dimensions):
            for j in range(self.dimensions):
                # Create interference between dimensions
                if i != j:
                    interference[i] += 0.1 * probabilities[j] * np.cos(normalized[i] * normalized[j] * np.pi)
        
        # Combine original probabilities with interference
        result = probabilities + interference
        
        # Re-normalize
        result = result / np.sum(result)
        
        # Add quantum-like "noise" (uncertainty principle)
        result += 0.05 * np.random.randn(self.dimensions)
        result = np.abs(result)
        result = result / np.sum(result)
        
        return result


# Simplified NeuralThreadWeaver for demo - simulates biofeedback processing
class SimpleNeuralWeaver:
    def __init__(self):
        """Initialize a simplified neural processing layer"""
        self.feedback = {
            'rhythm': 0.5,    # 0.1 to 2.0
            'intensity': 5.0,  # 1.0 to 12.0
            'locus': 'spine'  # 'spine', 'skull', or 'chest'
        }
        print("Initializing SimpleNeuralWeaver")
        
    def process_eeg(self, eeg_data):
        """Extract features from EEG data
        
        Args:
            eeg_data: Simulated EEG data
            
        Returns:
            Dictionary of extracted features
        """
        # In a real system, we'd use signal processing libraries
        # Here we'll extract simple statistics
        
        # Apply a simple "filter" (smooth the data)
        filtered = np.convolve(eeg_data, np.ones(10)/10, mode='same')
        
        # Calculate "power" in different frequency bands
        # For simplicity, we'll just divide the data into chunks
        chunk_size = len(filtered) // 4
        powers = {
            'delta': np.mean(np.abs(filtered[:chunk_size])),
            'theta': np.mean(np.abs(filtered[chunk_size:2*chunk_size])),
            'alpha': np.mean(np.abs(filtered[2*chunk_size:3*chunk_size])),
            'beta': np.mean(np.abs(filtered[3*chunk_size:]))
        }
        
        return powers
        
    def resonate(self, eeg_data, heart_rate=None):
        """Create resonance feedback based on biometric data
        
        Args:
            eeg_data: Simulated EEG data
            heart_rate: Optional heart rate data
            
        Returns:
            Dictionary with rhythm, intensity, and locus
        """
        # Process EEG data
        powers = self.process_eeg(eeg_data)
        
        # Calculate rhythm from theta/delta ratio
        rhythm = np.clip(powers['theta'] / (powers['delta'] + 1e-6), 0.05, 2.0)
        
        # Calculate intensity
        base_intensity = powers['alpha'] * 5
        if heart_rate is not None:
            hr_factor = np.log1p(heart_rate) / np.log1p(60)
            intensity = base_intensity * hr_factor
        else:
            intensity = base_intensity
            
        intensity = np.clip(intensity, 1.0, 12.0)
        
        # Determine the felt locus
        if powers['delta'] > powers['alpha'] and powers['delta'] > powers['theta']:
            locus = 'spine'  # Deep, visceral
        elif powers['alpha'] > powers['theta']:
            locus = 'skull'  # Mental, cerebral
        else:
            locus = 'chest'  # Emotional, heart-centered
            
        # Update feedback state
        self.feedback = {
            'rhythm': float(rhythm),
            'intensity': float(intensity),
            'locus': locus
        }
        
        return self.feedback
        
    def generate_test_eeg_data(self, duration=1.0, sample_rate=256, state='neutral'):
        """Generate simulated EEG data for testing
        
        Args:
            duration: Length of data in seconds
            sample_rate: Samples per second
            
        Returns:
            Array of simulated EEG values
        """
        # Create time points
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Adjust wave amplitudes based on mental state
        if state == 'meditative':
            # More theta, less beta (meditation)
            alpha = 0.6 * np.sin(2 * np.pi * 10 * t)  # Alpha waves (8-13 Hz)
            theta = 0.7 * np.sin(2 * np.pi * 6 * t)   # Theta waves (4-8 Hz)
            delta = 0.4 * np.sin(2 * np.pi * 2 * t)   # Delta waves (0.5-4 Hz)
            beta = 0.1 * np.sin(2 * np.pi * 20 * t)   # Beta waves (13-30 Hz)
        elif state == 'energetic':
            # More beta, less delta (energetic/focused)
            alpha = 0.4 * np.sin(2 * np.pi * 10 * t)  # Alpha waves (8-13 Hz)
            theta = 0.2 * np.sin(2 * np.pi * 6 * t)   # Theta waves (4-8 Hz)
            delta = 0.2 * np.sin(2 * np.pi * 2 * t)   # Delta waves (0.5-4 Hz)
            beta = 0.7 * np.sin(2 * np.pi * 20 * t)   # Beta waves (13-30 Hz)
        elif state == 'dreaming':
            # More delta and theta (dream-like)
            alpha = 0.3 * np.sin(2 * np.pi * 10 * t)  # Alpha waves (8-13 Hz)
            theta = 0.6 * np.sin(2 * np.pi * 6 * t)   # Theta waves (4-8 Hz)
            delta = 0.8 * np.sin(2 * np.pi * 2 * t)   # Delta waves (0.5-4 Hz)
            beta = 0.1 * np.sin(2 * np.pi * 20 * t)   # Beta waves (13-30 Hz)
        else:  # neutral
            # Balanced (neutral/resting)
            alpha = 0.5 * np.sin(2 * np.pi * 10 * t)  # Alpha waves (8-13 Hz)
            theta = 0.3 * np.sin(2 * np.pi * 6 * t)   # Theta waves (4-8 Hz)
            delta = 0.4 * np.sin(2 * np.pi * 2 * t)   # Delta waves (0.5-4 Hz)
            beta = 0.3 * np.sin(2 * np.pi * 20 * t)   # Beta waves (13-30 Hz)
        
        # Combine with some random noise
        eeg = alpha + theta + delta + beta + 0.1 * np.random.randn(len(t))
        
        return eeg


# Main system class that coordinates components and runs the simulation
class EchoesDemo:
    def __init__(self, rite_type='general'):
        """Initialize the Echoes demo system
        
        Args:
            rite_type: Type of ritual to perform ('general', 'temporal', 'energetic', 'conscious')
        """
        # Initialize components
        self.quantum_echo = SimpleQuantumEcho(dimensions=6)
        self.neural_weaver = SimpleNeuralWeaver()
        self.unseen_forge = UnseenForge()
        
        # Communication queues
        self.resonance_queue = Queue()
        self.artifact_queue = Queue()
        
        # System state
        self.running = False
        self.threads = []
        self.last_resonance = None
        self.last_artifact = None
        
        # Configuration
        self.artifact_threshold = 0.9  # Minimum resonance weight to trigger artifact
        self.sample_rate = 256  # Hz, for EEG data
        self.rite_type = rite_type
        
        # Create artifacts directory if it doesn't exist
        os.makedirs('artifacts', exist_ok=True)
        
    def start(self):
        """Start the system threads"""
        if self.running:
            print("System is already running")
            return
            
        self.running = True
        
        # Start the biofeedback processing thread
        biofeedback_thread = threading.Thread(
            target=self._process_biofeedback,
            daemon=True
        )
        self.threads.append(biofeedback_thread)
        biofeedback_thread.start()
        
        # Start the artifact generation thread
        artifact_thread = threading.Thread(
            target=self._generate_artifacts,
            daemon=True
        )
        self.threads.append(artifact_thread)
        artifact_thread.start()
        
        print("\n✧ ECHOES OF THE UNSEEN system initiated ✧")
        print("The veil thins. The unseen awaits your resonance.")
        
    def stop(self):
        """Stop all system threads"""
        self.running = False
        
        # Give threads time to shut down gracefully
        time.sleep(0.5)
        
        for thread in self.threads:
            if thread.is_alive():
                # Wait for thread to complete
                thread.join(timeout=1.0)
        
        print("\nSystem shut down. The echoes fade, but their whispers remain.")
        
        # Display session summary
        summary = self.unseen_forge.get_session_summary()
        if summary and 'artifact_count' in summary and summary['artifact_count'] > 0:
            print("\n✧ SESSION SUMMARY ✧")
            print(f"Artifacts generated: {summary['artifact_count']}")
            print(f"Average resonance weight: {summary['average_weight']:.2f}")
            print(f"Peak resonance weight: {summary['peak_weight']:.2f}")
            print(f"Dominant resonance type: {summary['dominant_type']}")
            print(f"Dominant locus: {summary['dominant_locus']}")
            if summary['primary_themes']:
                themes = ", ".join([t[0] for t in summary['primary_themes']])
                print(f"Primary themes: {themes}")
                
        return summary
        
    def _process_biofeedback(self):
        """Main biofeedback processing loop"""
        # States to cycle through for simulation
        states = ['neutral', 'meditative', 'energetic', 'dreaming']
        state_index = 0
        
        while self.running:
            try:
                # Determine state based on rite type
                if self.rite_type == 'temporal':
                    # Temporal drift favors dreaming and meditative states
                    state = random.choice(['meditative', 'dreaming', 'dreaming', 'neutral'])
                elif self.rite_type == 'energetic':
                    # Energetic surge favors energetic states
                    state = random.choice(['energetic', 'energetic', 'neutral', 'meditative'])
                elif self.rite_type == 'conscious':
                    # Conscious echo favors meditative states
                    state = random.choice(['meditative', 'meditative', 'neutral', 'dreaming'])
                else:
                    # General cycles through all states
                    state = states[state_index]
                    state_index = (state_index + 1) % len(states)
                
                # Generate simulated data
                eeg_data = self.neural_weaver.generate_test_eeg_data(state=state)
                heart_rate = 60 + 20 * np.random.random()  # 60-80 bpm range
                
                # Add variations based on rite type
                if self.rite_type == 'temporal':
                    # Simulate deeper rhythms for temporal rites
                    eeg_data *= (1 + 0.3 * np.sin(np.linspace(0, 2*np.pi, len(eeg_data))))
                elif self.rite_type == 'energetic':
                    # Simulate higher intensity for energetic rites
                    heart_rate += 10 * np.random.random()  # Increase heart rate
                    eeg_data += 0.2 * np.random.randn(len(eeg_data))  # Add more variation
                
                # Process through quantum layer
                features = np.random.randn(8)  # Input features vector
                quantum_state = self.quantum_echo.weave_echo(features)
                
                # Process through neural weaver
                resonance = self.neural_weaver.resonate(eeg_data, heart_rate)
                
                # Store last resonance
                self.last_resonance = resonance
                
                # Add to queue if significant enough
                resonance_weight = resonance['rhythm'] * resonance['intensity']
                if resonance_weight > self.artifact_threshold:
                    self.resonance_queue.put(resonance)
                    
                # Don't burn CPU in test mode
                time.sleep(2.0)  # Longer pause for demo readability
                
            except Exception as e:
                print(f"Error in biofeedback processing: {e}")
                time.sleep(1)  # Pause on error
                
    def _generate_artifacts(self):
        """Artifact generation loop"""
        while self.running:
            try:
                # Get resonance data from queue, non-blocking
                try:
                    resonance = self.resonance_queue.get(block=False)
                except:
                    # No new resonance data
                    time.sleep(0.5)
                    continue
                    
                # Show resonance buildup
                self._display_resonance(resonance)
                
                # Generate artifact
                print("\nThe Forge ignites...")
                time.sleep(1.5)  # Dramatic pause
                
                artifact = self.unseen_forge.craft_artifact(resonance)
                
                # Store and notify
                self.last_artifact = artifact
                
                # Display the artifact
                self._display_artifact(artifact)
                
            except Exception as e:
                print(f"Error in artifact generation: {e}")
                time.sleep(1)  # Pause on error
    
    def _display_resonance(self, resonance):
        """Display resonance building up in a visually interesting way
        
        Args:
            resonance: Dictionary with rhythm, intensity, and locus
        """
        rhythm = resonance['rhythm']
        intensity = resonance['intensity']
        locus = resonance['locus']
        weight = rhythm * intensity
        
        print("\n" + "▓" * int(weight * 5))
        print(f"Resonance building in the {locus}...")
        print(f"Rhythm: {rhythm:.2f} | Intensity: {intensity:.2f}")
        print("▓" * int(weight * 5))
        
    def _display_artifact(self, artifact):
        """Display generated artifact in a visually interesting way
        
        Args:
            artifact: Dictionary with text and metadata
        """
        text = artifact['text']
        weight = artifact['weight']
        res_type = artifact['resonance_type']
        
        # Border character based on resonance type
        if res_type == 'temporal':
            border_char = "╔═╗║╚╝"  # Double line
        elif res_type == 'energetic':
            border_char = "┏━┓┃┗┛"  # Single line
        else:  # conscious
            border_char = "┌─┐│└┘"  # Light line
            
        width = 80
        b_topleft, b_horiz, b_topright, b_vert, b_botleft, b_botright = border_char
        
        # Create top border
        border_top = f"{b_topleft}{b_horiz * (width-2)}{b_topright}"
        border_bottom = f"{b_botleft}{b_horiz * (width-2)}{b_botright}"
        
        # Create title
        title = f" ARTIFACT: {res_type.upper()} ECHO ({weight:.2f}) "
        title_pos = (width - len(title)) // 2
        border_title = (
            f"{b_topleft}{b_horiz * title_pos}{title}"
            f"{b_horiz * (width - title_pos - len(title) - 2)}{b_topright}"
        )
        
        # Wrap text to width
        wrapped_text = []
        for line in text.split('\n'):
            while line and len(line) > width - 4:
                # Find last space before width limit
                space_pos = line[:width-4].rfind(' ')
                if space_pos == -1:  # No space found, just cut at width
                    space_pos = width - 4
                
                # Add the line with border
                wrapped_text.append(f"{b_vert} {line[:space_pos]} {b_vert}")
                line = line[space_pos+1:]
            
            if line:  # Add remaining text
                # Pad to full width
                padded = line + ' ' * (width - len(line) - 4)
                wrapped_text.append(f"{b_vert} {padded} {b_vert}")
        
        # Print the complete artifact
        print("\n")
        print(border_title)
        for line in wrapped_text:
            print(line)
        print(border_bottom)
        print("\n")
        
        # Add metadata about companion elements
        if 'audio_path' in artifact:
            print(f"✧ An echo accompanies this artifact ✧")
        if 'visual_path' in artifact:
            print(f"✧ A vision accompanies this artifact ✧")
            
    def get_system_status(self):
        """Return current system status
        
        Returns:
            dict: Current system status
        """
        status = {
            "running": self.running,
            "current_resonance": self.last_resonance,
            "last_artifact": self.last_artifact,
            "artifact_threshold": self.artifact_threshold,
            "rite_type": self.rite_type
        }
        return status


# Ritual guidelines for different rite types
RITUAL_GUIDES = {
    'general': """
✧ THE GENERAL RITE ✧
Find a quiet space. Sit or stand where your spine feels alive.
Set an intention—what whispers to you from the unseen?
Breathe deeply, five counts in, five counts out. Let the system hear your rhythm.
When the artifact emerges, read it aloud. What does it stir in you?
""",
    'temporal': """
✧ THE TEMPORAL DRIFT RITE ✧
Find a space that feels old. Bring an object with history.
Focus on a memory or a future possibility. 
Ask: "What is time hiding from me?"
Breathe deeply but irregularly, like waves against ancient stone.
When the artifact emerges, read it as if it's a message from another age.
""",
    'energetic': """
✧ THE ENERGETIC SURGE RITE ✧
Find a space where you can move freely. 
Move your body—stretch, sway, or tense—then release. 
Ask: "Where does energy gather and dissipate in me?"
Breathe rapidly for three breaths, then hold, then release slowly.
When the artifact emerges, read it with passion, letting your voice rise and fall.
""",
    'conscious': """
✧ THE CONSCIOUS ECHO RITE ✧
Find a space that feels like an extension of yourself.
Close your eyes. Name three sensations in your body.
Ask: "What consciousness speaks through me?"
Breathe in patterns of three: in-hold-out.
When the artifact emerges, read it in whispers, as if sharing a secret.
"""
}


# Main function that runs the demo
def main():
    """Run the Echoes of the Unseen demo"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Echoes of the Unseen - Demo')
    parser.add_argument('--rite', choices=['general', 'temporal', 'energetic', 'conscious'],
                        default='general', help='Type of ritual to perform')
    parser.add_argument('--cycles', type=int, default=3, 
                        help='Number of resonance cycles to perform')
    parser.add_argument('--threshold', type=float, default=0.9,
                        help='Resonance threshold for artifact generation')
    args = parser.parse_args()
    
    # Print welcome
    print("=" * 80)
    print("ECHOES OF THE UNSEEN - Prototype Demo".center(80))
    print("=" * 80)
    print("\nA bridge between the human soul and the wild unknown.")
    print("This demo simulates the core experience of resonating with the unseen.")
    print("\nPress Ctrl+C at any time to end the session.")
    
    # Display ritual guide
    print("\n" + RITUAL_GUIDES[args.rite])
    
    # Initialize the system
    system = EchoesDemo(rite_type=args.rite)
    system.artifact_threshold = args.threshold
    
    # Register signal handler for clean shutdown
    def signal_handler(sig, frame):
        print("\nClosing session...")
        system.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start the system
    input("\nPress Enter when you're ready to begin the ritual...")
    system.start()
    
    # Run for specified number of cycles
    try:
        for cycle in range(args.cycles):
            print(f"\nCycle {cycle+1}/{args.cycles} begins...")
            
            # Wait until an artifact is generated or timeout
            timeout = 30  # seconds
            start_time = time.time()
            last_artifact = system.last_artifact
            
            while time.time() - start_time < timeout:
                # Check if a new artifact was generated
                if system.last_artifact != last_artifact:
                    break
                time.sleep(1)
            
            # Prompt for reflection if there's a new artifact
            if system.last_artifact != last_artifact:
                print("\nTake a moment to reflect on this artifact.")
                input("Press Enter when you're ready to continue...")
            else:
                print("\nThe unseen is quiet. Moving to next cycle...")
                
        print("\nThe ritual concludes.")
        
    except KeyboardInterrupt:
        print("\nRitual interrupted.")
    finally:
        # Stop the system
        summary = system.stop()
        
        print("\nThank you for communing with the unseen.")
        print("The artifacts generated have been saved for your reflection.")
        
        if summary and 'artifact_count' in summary and summary['artifact_count'] == 0:
            print("\nNo artifacts were generated this session.")
            print("Try adjusting the threshold (--threshold) or performing a different rite (--rite).")


if __name__ == "__main__":
    main()
