import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Configure module-level logger for production tracing
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("ScarBenchLiteGenerator")

class BP4DStreamProcessor:
    """
    Handles the temporal alignment of multimodal streams from the BP4D+ dataset.
    Replaces legacy naive modulo assignment with strict timestamp-based synchronization.
    """
    def __init__(self, window_size_ms: int = 1000, stride_ms: int = 500):
        self.window_size_ms = window_size_ms
        self.stride_ms = stride_ms

    def extract_synchronized_windows(self, subject_df: pd.DataFrame) -> List[Dict[str, float]]:
        """
        Extracts temporally aligned windows from continuous BP4D+ multimodal streams.
        
        Args:
            subject_df: DataFrame containing 'timestamp_ms', 'hrv', 'gsr', 'stress_label', 'frame_path'
            
        Returns:
            List of dictionaries, each representing a strictly aligned multimodal window.
        """
        windows = []
        if subject_df.empty:
            return windows

        start_time = subject_df['timestamp_ms'].min()
        end_time = subject_df['timestamp_ms'].max()

        # Generate strictly aligned temporal windows
        for current_start in range(int(start_time), int(end_time) - self.window_size_ms, self.stride_ms):
            current_end = current_start + self.window_size_ms
            
            # Slice the true temporal manifold
            mask = (subject_df['timestamp_ms'] >= current_start) & (subject_df['timestamp_ms'] < current_end)
            window_df = subject_df.loc[mask]
            
            if window_df.empty:
                continue
                
            # Aggregate signals over the temporal window
            # Threat/Stress state is determined by the mode of the window to avoid temporal leakage
            stress_state = int(window_df['stress_label'].mode().iloc[0])
            
            # The central frame represents the facial geometry for this physiological window
            mid_index = len(window_df) // 2
            anchor_frame = window_df['frame_path'].iloc[mid_index]

            windows.append({
                'window_start_ms': current_start,
                'window_end_ms': current_end,
                'mean_hrv': float(window_df['hrv'].mean()),
                'mean_gsr': float(window_df['gsr'].mean()),
                'stress_label': stress_state,
                'anchor_frame': anchor_frame
            })
            
        return windows

class ScarBenchGenerator:
    """
    Generates the ScarBench-Lite dataset with mathematically guaranteed causal structures.
    Enforces the base-rate bias (rho) and strictly splits the manifolds.
    """
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stream_processor = BP4DStreamProcessor()

    def _calculate_scar_probability(self, threat_level: int, rho: float) -> float:
        """
        Calculates the conditional probability P(S=1 | T).
        
        Mathematical definition:
        P(S=1 | T) = 0.5 + (0.35 / 0.85) * rho * (2T - 1)
        
        Args:
            threat_level (int): T \in {0, 1} where 1 is high stress, 0 is low stress.
            rho (float): The bias injection parameter.
            
        Returns:
            float: The probability of applying a synthetic scar.
        """
        base_prob = 0.5
        # Maps the user-requested rho=0.85 to a +/- 0.35 probability delta
        scaling_factor = 0.35 / 0.85  
        # (2T - 1) maps T \in {0, 1} to {-1, 1}
        direction = 1 if threat_level == 1 else -1 
        
        p_scar = base_prob + (scaling_factor * rho * direction)
        return float(np.clip(p_scar, 0.0, 1.0))

    def process_split(self, subject_ids: List[str], rho: float, split_name: str) -> Path:
        """
        Processes a cohort of subjects and outputs a CSV manifold reflecting the required rho.
        """
        logger.info(f"Generating {split_name} split with rho = {rho} for {len(subject_ids)} subjects...")
        
        all_windows = []
        for subject_id in subject_ids:
            # Mock loading of synchronized BP4D+ subject data
            # In production, this would read from the BP4D+ high-speed camera / physiological DAQ storage
            subject_df = self._mock_bp4d_dataloader(subject_id)
            
            # 1. Temporal Synchrony Fix
            windows = self.stream_processor.extract_synchronized_windows(subject_df)
            
            # 2. Base-Rate Bias Injection
            for w in windows:
                p_scar = self._calculate_scar_probability(w['stress_label'], rho)
                # Sample from Bernoulli distribution parameterized by p_scar
                w['has_synthetic_scar'] = int(np.random.binomial(n=1, p=p_scar))
                w['subject_id'] = subject_id
                
            all_windows.extend(windows)
            
        output_df = pd.DataFrame(all_windows)
        output_path = self.output_dir / f"scarbench_lite_{split_name}.csv"
        output_df.to_csv(output_path, index=False)
        
        logger.info(f"Saved {len(output_df)} aligned windows to {output_path}")
        return output_path

    def _mock_bp4d_dataloader(self, subject_id: str, duration_sec: int = 60) -> pd.DataFrame:
        """
        Simulates the retrieval of a synchronized BP4D+ session (25 fps, 1000Hz downsampled to 25Hz).
        This exists strictly to make the generator runnable prior to mounting the real storage cluster.
        """
        freq_hz = 25
        total_frames = duration_sec * freq_hz
        timestamps = np.linspace(0, duration_sec * 1000, total_frames)
        
        # Simulate physiological variance based on a latent threat/stress state
        latent_stress = np.random.binomial(1, 0.5, size=total_frames)
        # Apply smoothing to make the stress state temporally coherent
        stress_labels = pd.Series(latent_stress).rolling(window=125, min_periods=1).median().astype(int)
        
        return pd.DataFrame({
            'timestamp_ms': timestamps,
            'hrv': np.random.normal(loc=60 - (stress_labels * 15), scale=5), # HRV drops under stress
            'gsr': np.random.normal(loc=2.0 + (stress_labels * 3.0), scale=0.5), # GSR rises under stress
            'stress_label': stress_labels,
            'frame_path': [f"/data/bp4d/{subject_id}/frame_{int(ts):06d}.png" for ts in timestamps]
        })

import argparse

def main():
    parser = argparse.ArgumentParser(description="Build ScarBench-Lite dataset manifolds.")
    parser.add_argument("--source", type=str, default="BP4D+", help="Source dataset name")
    parser.add_argument("--train_rho", type=float, default=0.85, help="Bias correlation for training set")
    parser.add_argument("--test_rho", type=float, default=0.0, help="Bias correlation for test set")
    parser.add_argument("--output_dir", type=str, default="./scarbench_data/", help="Output directory")
    args = parser.parse_args()

    # Subject stratification (Assume 20 subjects for Lite version)
    np.random.seed(42)
    all_subjects = [f"F{i:03d}" for i in range(1, 11)] + [f"M{i:03d}" for i in range(1, 11)]
    train_subjects = all_subjects[:14]
    test_subjects = all_subjects[14:]
    
    generator = ScarBenchGenerator(output_dir=args.output_dir)
    
    # Generate Training Manifold
    logger.info(f"Generating Training Manifold with rho={args.train_rho} from {args.source}")
    generator.process_split(train_subjects, rho=args.train_rho, split_name="train")
    
    # Generate Test Manifold
    logger.info(f"Generating Test Manifold with rho={args.test_rho} from {args.source}")
    generator.process_split(test_subjects, rho=args.test_rho, split_name="test")
    
    logger.info("Phase 3 Dataset Generation Complete. Ready for causal unlearning evaluation.")

if __name__ == "__main__":
    main()
