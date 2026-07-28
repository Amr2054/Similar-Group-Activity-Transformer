"""Final tensor assembly: strict agent-slot ordering (ball, home, away) and saving to disk."""
import os
import numpy as np
import torch
from .constants import FEATURE_COLS, ROLE_HOME, ROLE_AWAY, ROLE_BALL


class TensorExportMixin:
    def save_to_tensor(self):
        """
        Imputes missing physical data, shapes the sequence into a strict [100, 23, 7] tensor,
        and saves it to disk as a compiled PyTorch binary.
        """
        print("Phase 5b: Compiling PyTorch Tensors")

        # 1. IMPUTE MISSING DATA (The NaN Guardrail)
        fill_cols = ['x', 'y', 'z', 'speed', 'dist_to_goal']
        self.final_extracted_df[fill_cols] = self.final_extracted_df.groupby(
            ['seq_id', 'role', 'player_id']
        )[fill_cols].transform(lambda x: x.ffill().bfill())

        # Safely fill any remaining NaNs (e.g., if a player was entirely missing for a whole sequence) with 0.0
        self.final_extracted_df[fill_cols] = self.final_extracted_df[fill_cols].fillna(0.0)

        # 2. PREPARE TENSOR DIMENSIONS
        sequences = self.final_extracted_df['seq_id'].unique()
        feature_cols = FEATURE_COLS
        X_list, labels_list, seq_ids_list = [], [], []

        for seq in sequences:
            seq_df = self.final_extracted_df[self.final_extracted_df['seq_id'] == seq]

            # Sort frames chronologically
            frames = np.sort(seq_df['videoTimeMs'].unique())

            # Ensure strict 100-frame enforcement from the downsampler
            if len(frames) != self.sample_size:
                print(f"  -> WARNING Seq {seq}: expected {self.sample_size} frames, got {len(frames)}. "
                      f"This should not happen after interpolate_to_fixed_length() — investigate.")
                continue

            # Initialize empty tensor for this specific play: [100, 23, 7]
            seq_tensor = np.zeros((self.sample_size, 23, len(feature_cols)), dtype=np.float32)

            for t_idx, t in enumerate(frames):
                frame_df = seq_df[seq_df['videoTimeMs'] == t]

                # --- AGENT ORDERING ENFORCEMENT ---

                # A. The Ball (Role 2) -> Always Index 0
                ball = frame_df[frame_df['role'] == ROLE_BALL]
                if not ball.empty:
                    seq_tensor[t_idx, 0, :] = ball[feature_cols].values[0]

                # B. Home Players (Role 0) -> Always Indices 1 through 11
                home = frame_df[frame_df['role'] == ROLE_HOME].sort_values('player_id')
                num_home = min(len(home), 11)  # Hard cap at 11 to prevent tensor shape errors
                if num_home > 0:
                    seq_tensor[t_idx, 1:1 + num_home, :] = home[feature_cols].values[:num_home]

                # C. Away Players (Role 1) -> Always Indices 12 through 22
                away = frame_df[frame_df['role'] == ROLE_AWAY].sort_values('player_id')
                num_away = min(len(away), 11)
                if num_away > 0:
                    seq_tensor[t_idx, 12:12 + num_away, :] = away[feature_cols].values[:num_away]

            X_list.append(seq_tensor)
            labels_list.append(seq_df['supcon_label'].iloc[0])
            seq_ids_list.append(seq)

        # 3. COMPILE AND SAVE
        if len(X_list) == 0:
            print("No valid sequences found to save.")
            return

        # Stack into final shape: [Num_Sequences, 100, 23, 7]
        X_final = torch.tensor(np.array(X_list), dtype=torch.float32)

        save_dict = {
            'features': X_final,
            'labels': labels_list,
            'sequence_ids': seq_ids_list
        }

        save_dir = f'{self.folder_path}/{self.save_folder}'
        os.makedirs(save_dir, exist_ok=True)
        save_path = f'{save_dir}/{self.game_id}.pt'

        torch.save(save_dict, save_path)
        print(f"Successfully saved tensor {X_final.shape} to {save_path}")

    # Todo : manual sequence label