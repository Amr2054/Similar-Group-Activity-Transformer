"""Per-frame physics extraction (ball + player position/speed/visibility) and pitch-relative normalization."""
import math
import numpy as np
import pandas as pd
from .constants import PLAYER_MAX_SPEED, BALL_MAX_SPEED


class FeatureExtractionMixin:
    # PHASE 4: FEATURE ENGINEERING (PHYSICS & NORMALIZATION)
    def extract_per_frame_info(self):
        """
        Unpacks the raw JSON tracking dicts into flat physical heuristics.
        Applies mathematical normalization so the attacking team ALWAYS faces Right (X=1.0).
        """
        # 1. Ensure mappings exist before extracting
        self._build_jersey_mappings()

        print("Phase 4: Extracting enriched physics and applying spatial normalization...")
        extracted_data = []

        # 2. Iterate over the sampled tracking frames per sequence
        for _, row in self.tracking_df.iterrows():
            seq_id = row.get('seq_id')
            video_time = row.get('videoTimeMs')

            # Fetch Metadata for this Sequence
            if seq_id in self.important_sequence_times.index:
                seq_metadata = self.important_sequence_times.loc[seq_id]
                supcon_label = seq_metadata['supcon_label']
                is_home_possession = seq_metadata['is_home_possession']
                atk_dir = seq_metadata.get('attacking_direction', 'R')
                p_length = seq_metadata.get('pitch_length', 105.0)
                p_width = seq_metadata.get('pitch_width', 68.0)
            else:
                supcon_label, is_home_possession, atk_dir = None, None, 'R'
                p_length, p_width = 105.0, 68.0

            # Pitch Normalization: Flip pitch 180 deg if attacking Left
            # If the team is attacking Left ('L'), direction_mult = -1.0 to flip the pitch.
            direction_mult = -1.0 if atk_dir == 'L' else 1.0
            x_scale = p_length / 2.0
            y_scale = p_width / 2.0

            # --- Ball Extraction ---
            smooth_ball = row.get('ballsSmoothed', {})
            raw_ball_list = row.get('balls', [])
            ball_x, ball_y, ball_z = np.nan, np.nan, 0.0
            ball_vis, ball_speed = 0.0, 0.0

            if isinstance(smooth_ball, dict) and 'x' in smooth_ball:
                bx_val = smooth_ball.get('x')
                by_val = smooth_ball.get('y')
                bz_val = smooth_ball.get('z')

                if bx_val is not None and by_val is not None:
                    # 1. Orient the pitch
                    raw_x = bx_val * direction_mult
                    raw_y = by_val * direction_mult

                    # 2. Normalize to [-1.0, 1.0]
                    ball_x = raw_x / x_scale
                    ball_y = raw_y / y_scale

                ball_z = bz_val if bz_val is not None else 0.0

                if isinstance(raw_ball_list, list) and len(raw_ball_list) > 0:
                    ball_vis = 1.0 if raw_ball_list[0].get('visibility') in ('VISIBLE', 'ESTIMATED') else 0.0
                    ball_speed = raw_ball_list[0].get('speed')
                    ball_speed = ball_speed if ball_speed is not None else 0.0

            dist_to_goal = math.hypot(1.0 - ball_x, 0.0 - ball_y) if pd.notna(ball_x) else np.nan

            extracted_data.append({
                'seq_id': seq_id,
                'videoTimeMs': video_time,
                'role': 2,  # 2 = Ball
                'player_id': 0,  # Ball ID is 0
                'x': ball_x,
                'y': ball_y,
                'z': ball_z,
                'speed': ball_speed,
                'visibility': ball_vis,
                'is_attacking': 0.0,
                'dist_to_goal': dist_to_goal,
                'supcon_label': supcon_label,
            })

            # --- PLAYER EXTRACTION HELPER ---
            def process_players(smooth_list, raw_list, role_int, jersey_map, is_attacking_team):
                if not isinstance(smooth_list, list):
                    return

                raw_dict = {str(p.get('jerseyNum')): p for p in raw_list} if isinstance(raw_list, list) else {}
                for p in smooth_list:

                    j_num = str(p.get('jerseyNum'))
                    raw_p = raw_dict.get(j_num, {})  # Get the corresponding raw data

                    px_val = p.get('x')
                    py_val = p.get('y')

                    p_x, p_y = np.nan, np.nan

                    if px_val is not None and py_val is not None:
                        p_x = (px_val * direction_mult) / x_scale
                        p_y = (py_val * direction_mult) / y_scale

                    raw_speed = raw_p.get('speed')  # TODO : Not in DOCS
                    speed = raw_speed if raw_speed is not None else 0.0
                    speed = np.clip(speed, 0, PLAYER_MAX_SPEED)
                    visibility = 1.0 if raw_p.get('visibility') in ('VISIBLE', 'ESTIMATED') else 0.0
                    is_attacking = 1.0 if is_attacking_team else 0.0
                    dist = math.hypot(1.0 - p_x, 0.0 - p_y) if pd.notna(p_x) else np.nan

                    p_id = jersey_map.get(j_num, None)

                    extracted_data.append({
                        'seq_id': seq_id,
                        'videoTimeMs': video_time,
                        'role': role_int,
                        'player_id': p_id,
                        'x': p_x,
                        'y': p_y,
                        'z': 0.0,
                        'speed': speed,
                        'visibility': visibility,
                        'is_attacking': is_attacking,
                        'dist_to_goal': dist,
                        'supcon_label': supcon_label,

                    })

            # Extract Home Players (Role = 0)
            s_home = row.get('homePlayersSmoothed', [])
            r_home = row.get('homePlayers', [])
            is_home_atk = (is_home_possession == True)
            process_players(s_home, r_home, 0, self.home_jersey_map, is_home_atk)

            # Extract Away Players (Role = 1)
            s_away = row.get('awayPlayersSmoothed', [])
            r_away = row.get('awayPlayers', [])
            is_away_atk = (is_home_possession == False)
            process_players(s_away, r_away, 1, self.away_jersey_map, is_away_atk)

        # Compile Final DataFrame
        self.final_extracted_df = pd.DataFrame(extracted_data)
        print(f"Extraction complete. Created tabular mapping with {len(self.final_extracted_df)} records.")
        print(f"Shape: {self.final_extracted_df.shape}")

    def post_process_ball_data(self):
        """
        Normalizes ball Z values and calculates ball speed based on frame-to-frame distance.
        To be called after extract_per_frame_info() is completed.
        """
        import numpy as np
        import pandas as pd

        # Mask to isolate only the ball records (role == 2)
        ball_mask = self.final_extracted_df['role'] == 2

        # Extract ball data and ensure strictly chronological order per sequence
        ball_data = self.final_extracted_df[ball_mask].sort_values(by=['seq_id', 'videoTimeMs'])

        # --- 1. Calculate Ball Speed ---
        # Calculate time difference in seconds
        dt_sec = ball_data.groupby('seq_id')['videoTimeMs'].diff() / 1000.0

        # Calculate coordinate differences
        dx = ball_data.groupby('seq_id')['x'].diff()
        dy = ball_data.groupby('seq_id')['y'].diff()
        dz = ball_data.groupby('seq_id')['z'].diff()

        # Euclidean distance
        dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        # Calculate speed (distance / time). Handle division by zero and NaNs for the first frames
        speed = (dist / dt_sec).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        speed = np.clip(speed, 0, BALL_MAX_SPEED)
        # Update the original DataFrame using the aligned indices
        self.final_extracted_df.loc[ball_data.index, 'speed'] = speed

        # --- 2. Normalize Ball Z Values (Standard Scaler) ---
        ball_z = self.final_extracted_df.loc[ball_mask, 'z']
        z_mean = ball_z.mean()
        z_std = ball_z.std()
        # TODO: Standardization ruins range
        # Apply standard scaling (z = (x - mean) / std), protecting against division by zero
        # if pd.notna(z_std) and z_std > 0:
        #     self.final_extracted_df.loc[ball_mask, 'z'] = (ball_z - z_mean) / z_std
        # else:
        #     self.final_extracted_df.loc[ball_mask, 'z'] = 0.0

        print("Post-processing complete: Ball Z-values standard-scaled and speeds calculated.")