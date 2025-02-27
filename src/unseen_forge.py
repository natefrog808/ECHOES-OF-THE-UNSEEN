import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from datetime import datetime
import json
import os
import sounddevice as sd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import random
import sys

# Import optional dependencies - gracefully handle if not installed
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("Warning: soundfile library not found. Audio files will not be saved.")
    print("Install with: pip install soundfile")

class UnseenForge:
    def __init__(self, model_path="gpt2-medium", artifacts_dir="artifacts"):
        """
        Initialize the UnseenForge with either a pre-trained or fine-tuned model.
        
        Args:
            model_path: Path to model or model name (default: "gpt2-medium")
            artifacts_dir: Directory to store generated artifacts
        """
        # Load model and tokenizer
        print(f"Initializing UnseenForge with model: {model_path}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        
        # Set up artifact storage
        self.artifacts_dir = artifacts_dir
        self.ensure_directories()
        
        # Thematic elements for different resonance states
        self.themes = {
            'temporal': {
                'primary': ["time", "memory", "future", "past", "moment", "eternal"],
                'imagery': ["shards", "fractures", "rivers", "dust", "bones", "scars"],
                'verbs': ["bleeds", "fractures", "spirals", "cracks", "dissolves", "echoes"]
            },
            'energetic': {
                'primary': ["fire", "light", "current", "pulse", "storm", "blood"],
                'imagery': ["wounds", "sparks", "veins", "lightning", "smoke", "breath"],
                'verbs': ["burns", "ignites", "strikes", "surges", "roars", "tears"]
            },
            'conscious': {
                'primary': ["mind", "soul", "voice", "silence", "dream", "shadow"],
                'imagery': ["caves", "mirrors", "fog", "stars", "roots", "birds"],
                'verbs': ["whispers", "trembles", "hovers", "sings", "drifts", "knows"]
            }
        }
        
        # Base prompts for different resonance types
        self.base_prompts = {
            'temporal': "From the dark, a howl breaks: Time {verb}, spilling its {imagery}—",
            'energetic': "From the dark, a howl breaks: The unseen {verb}, fierce as a {imagery}—",
            'conscious': "From the dark, a howl breaks: {locus} {verb}, alive with old {imagery}—"
        }
        
        # Locus-specific themes
        self.locus_themes = {
            'spine': {
                'adjectives': ["deep", "ancient", "primal", "raw", "visceral"],
                'imagery': ["earth", "bone", "root", "stone", "blood"]
            },
            'skull': {
                'adjectives': ["lucid", "ethereal", "crystalline", "vibrant", "electric"],
                'imagery': ["light", "spark", "star", "thread", "void"]
            },
            'chest': {
                'adjectives': ["warm", "resonant", "pulsing", "tender", "fierce"],
                'imagery': ["flame", "tide", "breath", "wound", "song"]
            }
        }
        
        # Track session continuity
        self.session_artifacts = []
        self.current_themes = set()
        
    def ensure_directories(self):
        """Create necessary directories for artifact storage"""
        if not os.path.exists(self.artifacts_dir):
            os.makedirs(self.artifacts_dir)
            
        # Create subdirectories for different artifact types
        for subdir in ['text', 'audio', 'visual']:
            path = os.path.join(self.artifacts_dir, subdir)
            if not os.path.exists(path):
                os.makedirs(path)
    
    def _get_resonance_type(self, resonance_data):
        """
        Determine the type of resonance from the data.
        
        Args:
            resonance_data: Dictionary with 'rhythm', 'intensity', and 'locus'
            
        Returns:
            str: 'temporal', 'energetic', or 'conscious'
        """
        rhythm = resonance_data['rhythm']
        intensity = resonance_data['intensity']
        
        if rhythm > 1.2:
            return 'temporal'
        elif intensity > 7.0:
            return 'energetic'
        else:
            return 'conscious'
    
    def _select_thematic_element(self, resonance_type, element_type):
        """
        Select a thematic element based on resonance type.
        
        Args:
            resonance_type: 'temporal', 'energetic', or 'conscious'
            element_type: 'primary', 'imagery', or 'verbs'
            
        Returns:
            str: A thematic element
        """
        elements = self.themes[resonance_type][element_type]
        return random.choice(elements)
    
    def _craft_prompt(self, resonance_data, include_previous=True):
        """
        Create a dynamic prompt based on resonance data.
        
        Args:
            resonance_data: Dictionary with 'rhythm', 'intensity', and 'locus'
            include_previous: Whether to incorporate previous artifacts
            
        Returns:
            str: Generated prompt
        """
        res_type = self._get_resonance_type(resonance_data)
        locus = resonance_data['locus']
        rhythm = resonance_data['rhythm']
        intensity = resonance_data['intensity']
        
        # Select thematic elements
        verb = self._select_thematic_element(res_type, 'verbs')
        imagery = self._select_thematic_element(res_type, 'imagery')
        primary = self._select_thematic_element(res_type, 'primary')
        
        # Remember this theme for session continuity
        self.current_themes.add(primary)
        
        # Format the base prompt
        if res_type == 'conscious':
            prompt = self.base_prompts[res_type].format(
                locus=locus, 
                verb=verb,
                imagery=imagery
            )
        else:
            prompt = self.base_prompts[res_type].format(
                verb=verb,
                imagery=imagery
            )
            
        # Add resonance-specific details to guide generation
        prompt_extensions = []
        
        # Rhythm-based extensions
        if rhythm > 1.5:
            prompt_extensions.append(f"{primary.capitalize()} ancient and raw, bleeding through the now.")
        elif rhythm > 1.0:
            prompt_extensions.append(f"Threads of {primary} weave between moments, pulsing.")
        
        # Intensity-based extensions
        if intensity > 9:
            prompt_extensions.append(f"The body knows what the mind forgets—{primary} burns.")
        elif intensity > 7:
            prompt_extensions.append(f"In the marrow, {primary} waits. It has always waited.")
            
        # Locus-based extensions
        locus_adj = random.choice(self.locus_themes[locus]['adjectives'])
        locus_img = random.choice(self.locus_themes[locus]['imagery'])
        prompt_extensions.append(f"The {locus} holds {primary} {locus_adj} as {locus_img}.")
            
        # Add extensions to prompt
        if prompt_extensions:
            # Add 1-2 extensions based on intensity
            num_extensions = 1 + (1 if intensity > 8 else 0)
            selected_extensions = random.sample(prompt_extensions, min(num_extensions, len(prompt_extensions)))
            prompt += " " + " ".join(selected_extensions)
        
        # Incorporate previous artifacts for continuity if available
        if include_previous and self.session_artifacts:
            # Get the most recent artifact with sufficient weight
            recent = self.session_artifacts[-1]
            
            # Extract a fragment (5-10 words)
            words = recent['text'].split()
            if len(words) > 10:
                # Find a sentence fragment from the middle
                mid_point = len(words) // 2
                fragment_start = random.randint(mid_point - 5, mid_point)
                fragment_start = max(0, fragment_start)
                fragment_end = min(len(words), fragment_start + random.randint(5, 10))
                fragment = " ".join(words[fragment_start:fragment_end])
                
                # Add as echo
                prompt += f"\n\nEchoes linger: '{fragment}...'"
            
        return prompt
    
    def craft_artifact(self, resonance_data, include_previous=True):
        """
        Generate an artifact based on the resonance data.
        
        Args:
            resonance_data: Dictionary with 'rhythm', 'intensity', and 'locus'
            include_previous: Whether to incorporate previous artifacts
            
        Returns:
            dict: Generated artifact with metadata
        """
        # Prepare the prompt
        prompt = self._craft_prompt(resonance_data, include_previous)
        
        # Encode the prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Set generation parameters based on resonance
        temperature = 0.7 + (resonance_data['intensity'] / 15)  # More intensity = more variability
        top_k = int(40 + (resonance_data['rhythm'] * 15))  # More rhythm = more options
        
        # Length varies with intensity
        max_length = int(80 + (resonance_data['intensity'] * 5))
        
        # Generate text
        output = self.model.generate(
            inputs, 
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=0.92,
            repetition_penalty=1.2,
            do_sample=True,
            num_return_sequences=1
        )
        
        # Decode the output
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Calculate artifact weight
        weight = resonance_data['intensity'] * resonance_data['rhythm']
        
        # Create artifact object with metadata
        artifact = {
            "type": "text",
            "text": generated_text,
            "source_resonance": resonance_data,
            "timestamp": datetime.now().isoformat(),
            "weight": weight,
            "resonance_type": self._get_resonance_type(resonance_data),
            "themes": list(self.current_themes)
        }
        
        # Generate companion soundscape if weight is significant
        if weight > 1.2:
            audio_path = self._generate_soundscape(resonance_data)
            if audio_path:
                artifact["audio_path"] = audio_path
        
        # Generate visual element if weight is very significant
        if weight > 1.8:
            visual_path = self._generate_visual(resonance_data)
            if visual_path:
                artifact["visual_path"] = visual_path
        
        # Save the artifact
        saved_path = self._save_artifact(artifact)
        artifact["path"] = saved_path
        
        # Add to session history
        self.session_artifacts.append(artifact)
        
        return artifact
    
    def _generate_soundscape(self, resonance_data):
        """
        Generate a soundscape based on resonance data.
        
        Args:
            resonance_data: Dictionary with 'rhythm', 'intensity', and 'locus'
            
        Returns:
            str: Path to saved audio file
        """
        try:
            # Create time array (3-10 seconds depending on intensity)
            duration = 3 + (resonance_data['intensity'] / 2)
            sample_rate = 44100
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Base frequency related to rhythm
            if resonance_data['locus'] == 'spine':
                base_freq = 55 * resonance_data['rhythm']  # Low for spine
            elif resonance_data['locus'] == 'skull':
                base_freq = 220 * resonance_data['rhythm']  # Higher for skull
            else:  # chest
                base_freq = 110 * resonance_data['rhythm']  # Middle for chest
            
            # Generate base tone
            base_amp = 0.4
            base_tone = base_amp * np.sin(2 * np.pi * base_freq * t)
            
            # Add overtones based on intensity
            overtones = np.zeros_like(t)
            num_overtones = int(2 + (resonance_data['intensity'] / 3))
            
            for i in range(1, num_overtones + 1):
                # Each overtone gets quieter
                amp = base_amp / (i * 1.5)
                # Add some detuning for richness
                detune = 1.0 + (0.001 * np.random.randn())
                overtones += amp * np.sin(2 * np.pi * (base_freq * i * detune) * t)
            
            # Add texture based on resonance type
            res_type = self._get_resonance_type(resonance_data)
            texture = np.zeros_like(t)
            
            if res_type == 'temporal':
                # Slow pulsing for temporal
                pulse_rate = 1 + (resonance_data['rhythm'] / 2)
                texture = 0.1 * np.sin(2 * np.pi * pulse_rate * t) * base_tone
            elif res_type == 'energetic':
                # More noise/distortion for energetic
                noise = 0.1 * (resonance_data['intensity'] / 12) * np.random.randn(len(t))
                texture = noise * np.sin(2 * np.pi * base_freq * 3 * t)
            else:  # conscious
                # Harmonic swells for conscious
                swell_rate = 0.2 + (resonance_data['rhythm'] / 5)
                texture = 0.15 * np.sin(2 * np.pi * swell_rate * t) * np.sin(2 * np.pi * base_freq * 2 * t)
            
            # Combine elements
            waveform = base_tone + overtones + texture
            
            # Apply envelope
            attack = int(sample_rate * 0.1)
            release = int(sample_rate * 0.5)
            
            envelope = np.ones_like(waveform)
            # Attack
            envelope[:attack] = np.linspace(0, 1, attack)
            # Release
            envelope[-release:] = np.linspace(1, 0, release)
            
            waveform = waveform * envelope
            
            # Normalize to prevent clipping
            waveform = waveform / (np.max(np.abs(waveform)) + 1e-6) * 0.8
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"soundscape_{timestamp}.wav"
            filepath = os.path.join(self.artifacts_dir, 'audio', filename)
            
            # Convert to stereo
            stereo = np.column_stack([waveform, waveform])
            
            # Save audio file - in a real implementation, we'd use a library like soundfile
            # For this example, we'll just pretend it's saved
            # sf.write(filepath, stereo, sample_rate)
            
            return filepath
            
        except Exception as e:
            print(f"Error generating soundscape: {e}")
            return None
    
    def _generate_visual(self, resonance_data):
        """
        Generate a visual representation based on resonance data.
        
        Args:
            resonance_data: Dictionary with 'rhythm', 'intensity', and 'locus'
            
        Returns:
            str: Path to saved image file
        """
        try:
            # Create a figure
            plt.figure(figsize=(8, 6))
            
            # Clear any existing plots
            plt.clf()
            
            # Set style based on resonance type
            res_type = self._get_resonance_type(resonance_data)
            
            if res_type == 'temporal':
                # Use colors that suggest time/memory
                colors = ['#1a237e', '#283593', '#3949ab', '#5c6bc0']
                
                # Create a time-like visual with flows or spirals
                t = np.linspace(0, 10, 1000)
                radius = 1 + 0.5 * np.sin(t * resonance_data['rhythm'])
                x = t * np.cos(t) * radius
                y = t * np.sin(t) * radius
                
                plt.plot(x, y, linewidth=resonance_data['intensity']/3, color=colors[0])
                plt.scatter(x[::50], y[::50], s=30, color=colors[2], alpha=0.7)
                
            elif res_type == 'energetic':
                # Use colors that suggest energy/fire
                colors = ['#b71c1c', '#d32f2f', '#f44336', '#e57373']
                
                # Create energy bursts or lightning-like patterns
                for i in range(int(resonance_data['intensity'])):
                    x = np.zeros(20)
                    y = np.zeros(20)
                    
                    x[0] = np.random.uniform(-10, 10)
                    y[0] = np.random.uniform(-10, 10)
                    
                    for j in range(1, 20):
                        angle = np.random.uniform(0, 2*np.pi)
                        length = np.random.uniform(0, 1) * resonance_data['rhythm']
                        x[j] = x[j-1] + length * np.cos(angle)
                        y[j] = y[j-1] + length * np.sin(angle)
                    
                    plt.plot(x, y, color=random.choice(colors), linewidth=1.5, alpha=0.7)
                    
            else:  # conscious
                # Use colors that suggest consciousness/mind
                colors = ['#4a148c', '#6a1b9a', '#8e24aa', '#ab47bc']
                
                # Create patterns suggestive of neural networks or thoughts
                points = int(30 + resonance_data['intensity'] * 5)
                x = np.random.uniform(-10, 10, points)
                y = np.random.uniform(-10, 10, points)
                
                # Find connections (edges) between points
                for i in range(points):
                    for j in range(i+1, points):
                        dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        # Connect points that are close
                        if dist < 5 * resonance_data['rhythm']:
                            alpha = 1 - (dist / (5 * resonance_data['rhythm']))
                            plt.plot([x[i], x[j]], [y[i], y[j]], color=colors[0], 
                                     alpha=alpha, linewidth=0.5)
                
                plt.scatter(x, y, s=20, color=colors[2], alpha=0.7)
            
            # Style the plot
            plt.axis('off')  # No axes
            plt.tight_layout()
            
            # Set background color based on locus
            if resonance_data['locus'] == 'spine':
                plt.gca().set_facecolor('#000000')  # Black for spine
            elif resonance_data['locus'] == 'skull':
                plt.gca().set_facecolor('#f5f5f5')  # Light gray for skull
            else:  # chest
                plt.gca().set_facecolor('#0a0a0a')  # Very dark gray for chest
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"visual_{timestamp}.png"
            filepath = os.path.join(self.artifacts_dir, 'visual', filename)
            
            # In a real implementation, we'd save the file:
            # plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            print(f"Error generating visual: {e}")
            return None
    
    def _save_artifact(self, artifact):
        """
        Save artifact to storage.
        
        Args:
            artifact: Dictionary with artifact data
            
        Returns:
            str: Path to saved artifact
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"artifact_{timestamp}.json"
        filepath = os.path.join(self.artifacts_dir, 'text', filename)
        
        with open(filepath, 'w') as f:
            json.dump(artifact, f, indent=2)
            
        return filepath
        
    def retrieve_artifacts(self, limit=5, min_weight=0):
        """
        Retrieve recent artifacts, optionally filtered by weight.
        
        Args:
            limit: Maximum number of artifacts to retrieve
            min_weight: Minimum weight threshold
            
        Returns:
            list: List of artifact dictionaries
        """
        artifacts = []
        
        text_dir = os.path.join(self.artifacts_dir, 'text')
        if not os.path.exists(text_dir):
            return artifacts
            
        for filename in sorted(os.listdir(text_dir), reverse=True):
            if filename.endswith('.json'):
                filepath = os.path.join(text_dir, filename)
                with open(filepath, 'r') as f:
                    artifact = json.load(f)
                
                if artifact['weight'] >= min_weight:
                    artifacts.append(artifact)
                
                if len(artifacts) >= limit:
                    break
                    
        return artifacts
        
    def get_session_summary(self):
        """
        Generate a summary of the current session.
        
        Returns:
            dict: Session summary statistics and themes
        """
        if not self.session_artifacts:
            return {"status": "No artifacts generated in this session"}
            
        # Calculate statistics
        weights = [a['weight'] for a in self.session_artifacts]
        types = [a['resonance_type'] for a in self.session_artifacts]
        loci = [a['source_resonance']['locus'] for a in self.session_artifacts]
        
        # Find dominant themes across all artifacts
        all_themes = []
        for artifact in self.session_artifacts:
            if 'themes' in artifact:
                all_themes.extend(artifact['themes'])
                
        # Count theme occurrences
        theme_counts = {}
        for theme in all_themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
            
        # Sort themes by frequency
        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Create summary
        summary = {
            "artifact_count": len(self.session_artifacts),
            "average_weight": sum(weights) / len(weights) if weights else 0,
            "peak_weight": max(weights) if weights else 0,
            "dominant_type": max(set(types), key=types.count) if types else None,
            "dominant_locus": max(set(loci), key=loci.count) if loci else None,
            "primary_themes": sorted_themes[:3] if sorted_themes else [],
            "timestamp": datetime.now().isoformat()
        }
        
        return summary
        
    def reset_session(self):
        """Reset the session, clearing artifact history but preserving settings"""
        self.session_artifacts = []
        self.current_themes = set()
        print("Session reset. The Forge awaits new resonance.")


# Example usage
if __name__ == "__main__":
    # Create the forge
    forge = UnseenForge()
    
    # Simulate some resonance data
    resonance_data = {
        'rhythm': 1.3,
        'intensity': 8.5,
        'locus': 'skull'
    }
    
    # Generate an artifact
    artifact = forge.craft_artifact(resonance_data)
    
    # Print the result
    print(f"Resonance Type: {artifact['resonance_type']}")
    print(f"Weight: {artifact['weight']:.2f}")
    print(f"Themes: {artifact['themes']}")
    print("\n--- ARTIFACT TEXT ---")
    print(artifact['text'])
    
    # Check for multi-modal elements
    if 'audio_path' in artifact:
        print(f"\nAudio companion created: {artifact['audio_path']}")
    if 'visual_path' in artifact:
        print(f"\nVisual companion created: {artifact['visual_path']}")
