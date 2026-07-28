import bz2
import json
import math
import pandas as pd
import numpy as np
import torch
import os

class FIFAWC22:
    """
        End-to-End Data Engineering Pipeline for PFF Football Tracking Data.

        This class ingests raw PFF Event and Tracking JSONs, extracts tactical
        sequences (Shots, Crosses, Fouls), normalizes pitch coordinates, and
        compiles the physics into standard [100, 23, 7] PyTorch tensors
        ready for Supervised Contrastive Learning (SupCon).
        """

    # PHASE 1: INITIALIZATION & I/O
    def __init__(self, folder_path, game_id, sample_size=100, pre_buffer=10, post_buffer=3, save_Tensor = False, save_folder="Processed Tensors"):
        self.folder_path = folder_path
        self.game_id = game_id

        # Hyperparameters for tensor shaping
        self.sample_size = sample_size
        self.pre_buffer = pre_buffer
        self.post_buffer = post_buffer
        self.save_folder = save_folder
        # Execute the pipeline sequentially
        self.PLAYER_MAX_SPEED = 12.0
        self.BALL_MAX_SPEED = 35.0  # a well-struck shot can exceed 30 m/s; keep this well above player speed

        self.load_event_data()
        self.get_important_sequences()
        self.load_tracking_data()
        self.extract_per_frame_info()  # now reads ALL raw frames, not a pre-thinned subset
        self.post_process_ball_data()
        self.interpolate_to_fixed_length()  # NEW — replaces sample_sequence_frames()
        self.validate_extraction(sample_seq=10)
        if save_Tensor:
            self.save_to_tensor()
    
    def load_event_data(self):
        """Loads the raw Event JSON and calculates absolute sequence boundaries."""
        with open(f'{self.folder_path}/Event Data/{self.game_id}.json', 'rt') as f:
            events_data = json.load(f)
        self.events_df = pd.DataFrame(events_data)
        
        # Group by sequence to find global start/end times
        self.sequences = self.events_df.groupby('sequence').agg(
            start_time=('startTime', 'min'),
            end_time=('endTime', 'max')
        )
        print(f"Phase 1: Found {len(self.sequences)} valid sequences in Event Data.")

    # PHASE 2: EVENT DATA ENGINEERING (LABELS & WINDOWS)
    def get_important_sequences(self):
        """
        Identifies tactical anchors, chains compound labels (e.g., CR_I__SH_S),
        and calculates precise elastic tracking windows clamped to sequence boundaries.
        """
        target_event_types = {'SH', 'CR', 'FO'} # Updated focus
        seq_boundaries = {}
        raw_anchors = []

        # Step 1: Map global sequence boundaries to prevent turnover corruption
        for _, row in self.events_df.iterrows():
            seq_id = row.get('sequence')
            e_time = row.get('eventTime')
            if pd.notna(seq_id) and pd.notna(e_time):
                if seq_id not in seq_boundaries:
                    seq_boundaries[seq_id] = {'start': e_time, 'end': e_time}
                else:
                    seq_boundaries[seq_id]['start'] = min(seq_boundaries[seq_id]['start'], e_time)
                    seq_boundaries[seq_id]['end'] = max(seq_boundaries[seq_id]['end'], e_time)

        # Step 2: Extract anchors and dynamically build SupCon labels
        for _, row in self.events_df.iterrows():
            poss_event = row.get('possessionEvents')
            if not isinstance(poss_event, dict): 
                continue

            event_type = poss_event.get('possessionEventType')
            if event_type in target_event_types:
                seq_id = row.get('sequence')

                # Handle PFF JSON dictionary inconsistencies based on event type
                sub_type, outcome = None, None
                if event_type == 'SH':
                    sub_type = poss_event.get('shotType')
                    outcome = poss_event.get('shotOutcomeType')
                elif event_type == 'CR':
                    sub_type = poss_event.get('crossType')
                    outcome = poss_event.get('crossOutcomeType')
                elif event_type == 'FO':
                    foul_data = row.get('fouls', {})
                    if isinstance(foul_data, dict):
                        sub_type = foul_data.get('finalOffenseType')
                        outcome = foul_data.get('finalFoulOutcomeType')

                # Build single event label
                label_parts = [event_type]
                if sub_type: label_parts.append(sub_type)
                if outcome: label_parts.append(outcome)
                single_label = "_".join(label_parts)

                # Extract pitch dimensions and attacking direction for later normalization
                game_events = row.get('gameEvents', {})
                is_home_possession = game_events.get('homeTeam') if isinstance(game_events, dict) else None

                stadium_meta = row.get('stadiumMetadata', {})
                atk_dir, pitch_length, pitch_width = 'R', 105.0, 68.0
                if isinstance(stadium_meta, dict):
                    atk_dir = self._safe_val(stadium_meta.get('teamAttackingDirection'), 'R')
                    pitch_length = self._safe_val(stadium_meta.get('pitchLength'), 105.0)
                    pitch_width = self._safe_val(stadium_meta.get('pitchWidth'), 68.0)

                raw_anchors.append({
                    'sequence_id': seq_id,
                    'event_time': row.get('eventTime'),
                    'single_label': single_label,
                    'is_home_possession': is_home_possession,
                    'attacking_direction': atk_dir,
                    'pitch_length': pitch_length,
                    'pitch_width': pitch_width,
                    'seq_start': seq_boundaries[seq_id]['start'],
                    'seq_end': seq_boundaries[seq_id]['end']
                })

        # Step 3: Chronological Chaining & Elastic Windows
        sequences = {}
        for anchor in raw_anchors:
            seq_id = anchor['sequence_id']
            if seq_id not in sequences: 
                sequences[seq_id] = []
            sequences[seq_id].append(anchor)

        important_seqs_data = []
        for seq_id, events in sequences.items():
            events.sort(key=lambda x: x['event_time'])

            # Chain multiple events into one tactical string (e.g., CR_I_D __ SH_S_S)
            compound_label = " __ ".join([e['single_label'] for e in events])

            # Stretch window and clamp strictly to sequence boundaries
            first_time = events[0]['event_time']
            last_time = events[-1]['event_time']
            start_t = max(first_time - self.pre_buffer, events[0]['seq_start'])
            end_t = min(last_time + self.post_buffer, events[0]['seq_end'])

            important_seqs_data.append({
                'sequence_id': seq_id,
                'start_time': start_t,
                'end_time': end_t,
                'supcon_label': compound_label,
                'is_home_possession': events[0]['is_home_possession'],
                'attacking_direction':events[0]['attacking_direction'],
                'pitch_length':events[0]['pitch_length'],
                'pitch_width':events[0]['pitch_width'],
            })

        self.important_sequence_times = pd.DataFrame(important_seqs_data).set_index('sequence_id')
        print(f"Phase 2: Built {len(self.important_sequence_times)} Elastic Windows with Compound Labels.")

    # PHASE 3: TRACKING DATA ENGINEERING (EXTRACTION & SAMPLING)
    def load_tracking_data(self):
        """Streams the heavy .bz2 tracking file, extracting only frames inside our tactical windows."""

        print("Loading and filtering tracking data")
        self.tracking_frames = []
        file_path = f'{self.folder_path}/Tracking Data/{self.game_id}.jsonl.bz2'
        
        # Pre-extract intervals to a native Python list for O(1) loop lookups
        intervals = [
            (row.Index, row.start_time * 1000, row.end_time * 1000)
            for row in self.important_sequence_times.itertuples()
        ]
        
        with bz2.open(file_path, 'rt') as f:
            for line in f:
                if line.strip():
                    frame = json.loads(line)
                    t = frame['videoTimeMs']
                    seq_id = -1
                    
                    # Iterate over the pre-calculated list, not the pandas dataframe
                    for seq, start_ms, end_ms in intervals:
                        if start_ms <= t <= end_ms:
                            seq_id = int(seq)
                            break
                    
                    if seq_id != -1:
                        frame['seq_id'] = seq_id
                        self.tracking_frames.append(frame)
                
        # Sort tracking frames by video time to maintain chronological order
        self.tracking_frames = sorted(self.tracking_frames, key=lambda x: x['videoTimeMs'])
        self.tracking_df = pd.DataFrame(self.tracking_frames)
        print(f"  -> Extracted {len(self.tracking_df)} total tactical frames.")

    def sample_sequence_frames(self):
        """
        Calculates the target number of frames per unique event based on self.sample_size.
        Reallocates unused frame quotas from short events to longer events to ensure
        the final sequence hits the target sample_size.
        """

        print("Phase 3b: Balancing frame counts per sequence")
        sampled_frames_list = []

        for seq_id, seq_df in self.tracking_df.groupby('seq_id'):
            seq_df = seq_df.copy()

            # 1. Fill NaNs so we don't lose the continuous tracking data
            seq_df['game_event_id'] = seq_df['game_event_id'].ffill().bfill()

            unique_events = seq_df['game_event_id'].unique()
            num_events = len(unique_events)

            if num_events == 0:
                continue

            # Initial target per event
            base_target = self.sample_size // num_events
            remainder = self.sample_size % num_events

            targets = {}
            for i, ev_id in enumerate(unique_events):
                targets[ev_id] = base_target + (1 if i < remainder else 0)

            # 2. First pass: Check for "short" events and calculate the deficit
            deficit = 0
            for ev_id in unique_events:
                actual_frames = len(seq_df[seq_df['game_event_id'] == ev_id])
                if actual_frames < targets[ev_id]:
                    # We have fewer frames than the target. Calculate how many we are short.
                    deficit += (targets[ev_id] - actual_frames)
                    targets[ev_id] = actual_frames # Cap the target to what's available

            # 3. Reallocate the deficit to events that have extra frames
            if deficit > 0:
                for ev_id in unique_events:
                    actual_frames = len(seq_df[seq_df['game_event_id'] == ev_id])
                    extra_capacity = actual_frames - targets[ev_id]

                    if extra_capacity > 0:
                        # Give this event as much of the deficit as it can handle
                        added_frames = min(extra_capacity, deficit)
                        targets[ev_id] += added_frames
                        deficit -= added_frames

                    if deficit == 0:
                        break # All missing frames reallocated!

            # 4. Final Pass: Sample the frames using the updated targets
            for ev_id in unique_events:
                event_frames = seq_df[seq_df['game_event_id'] == ev_id]
                t_frames = targets[ev_id]
                a_frames = len(event_frames)

                if a_frames > t_frames:
                    indices = np.linspace(0, a_frames - 1, t_frames, dtype=int)
                    sampled_frames_list.append(event_frames.iloc[indices])
                else:
                    sampled_frames_list.append(event_frames)

        # Reconstruct the DataFrame and sort chronologically
        if sampled_frames_list:
            self.sampled_tracking_df = pd.concat(sampled_frames_list).sort_values(by=['seq_id', 'videoTimeMs']).reset_index(drop=True)
            print(f"Downsampling complete. New total frames: {len(self.sampled_tracking_df)}")
            print(f"Sequence 10 now has: {len(self.sampled_tracking_df[self.sampled_tracking_df['seq_id'] == 10])} frames")
        else:
            self.sampled_tracking_df = pd.DataFrame()
            print("Warning: No frames were left after downsampling.")

    def _build_jersey_mappings(self):
        """
        Builds jerseyNum -> playerId mapping for all players (including subs)
        without the heavy overhead of .iterrows().
        """
        print("Building Jersey-to-PlayerID mapping from Event Data...")
        self.home_jersey_map = {}
        self.away_jersey_map = {}
        
        # Fast iteration directly over the Series (ignores NaNs)
        for players_list in self.events_df['homePlayers'].dropna():
            for p in players_list:
                if 'jerseyNum' in p and 'playerId' in p:
                    self.home_jersey_map[str(p['jerseyNum'])] = p['playerId']
                    
        for players_list in self.events_df['awayPlayers'].dropna():
            for p in players_list:
                if 'jerseyNum' in p and 'playerId' in p:
                    self.away_jersey_map[str(p['jerseyNum'])] = p['playerId']

    @staticmethod
    def _safe_val(value, default):
        if value is None:
            return default
        try:
            if pd.isna(value):
                return default
        except (TypeError, ValueError):
            pass  # not NaN-checkable (e.g. a string) -> keep as-is
        return value

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
                    ball_vis = 1.0 if raw_ball_list[0].get('visibility') in ('VISIBLE','ESTIMATED') else 0.0
                    ball_speed = raw_ball_list[0].get('speed')
                    ball_speed = ball_speed if ball_speed is not None else 0.0

            dist_to_goal = math.hypot(1.0 - ball_x, 0.0 - ball_y) if pd.notna(ball_x) else np.nan

            extracted_data.append({
                'seq_id': seq_id,
                'videoTimeMs': video_time,
                'role': 2,        # 2 = Ball
                'player_id': 0,   # Ball ID is 0
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

                    raw_speed = raw_p.get('speed') # TODO : Not in DOCS
                    speed = raw_speed if raw_speed is not None else 0.0
                    speed = np.clip(speed, 0, self.PLAYER_MAX_SPEED)
                    visibility = 1.0 if raw_p.get('visibility') in ('VISIBLE','ESTIMATED') else 0.0
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
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Calculate speed (distance / time). Handle division by zero and NaNs for the first frames
        speed = (dist / dt_sec).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        speed = np.clip(speed, 0, self.BALL_MAX_SPEED)
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

    def interpolate_to_fixed_length(self):
        """
        Resamples every (seq_id, role, player_id) trajectory onto a shared
        grid of exactly `sample_size` evenly-spaced timestamps per sequence,
        via linear interpolation on real elapsed time.
        """
        print("Phase 4b: Interpolating all sequences onto a fixed time grid")
        interp_cols = ['x', 'y', 'z', 'speed', 'visibility', 'is_attacking', 'dist_to_goal']
        resampled_rows = []
        dropped_agents = 0

        for seq_id, seq_df in self.final_extracted_df.groupby('seq_id'):
            if seq_id not in self.important_sequence_times.index:
                continue
            meta = self.important_sequence_times.loc[seq_id]
            t_start, t_end = meta['start_time'] * 1000.0, meta['end_time'] * 1000.0
            if not (t_end > t_start):
                print(f"  -> Skipping Seq {seq_id}: degenerate window (start >= end).")
                continue

            target_times = np.linspace(t_start, t_end, self.sample_size)

            for (role, player_id), group in seq_df.groupby(['role', 'player_id']):
                valid = group.dropna(subset=['x', 'y']).sort_values('videoTimeMs')

                if len(valid) == 0:
                    dropped_agents += 1
                    continue  # never actually tracked in this window — correctly absent
                elif len(valid) == 1:
                    src_t = np.array([t_start, t_end])
                    src_vals = {c: np.repeat(valid[c].values, 2) for c in interp_cols}
                else:
                    src_t = valid['videoTimeMs'].values
                    src_vals = {c: valid[c].ffill().bfill().values for c in interp_cols}

                interp_feats = {c: np.interp(target_times, src_t, src_vals[c]) for c in interp_cols}

                resampled_rows.append(pd.DataFrame({
                    'seq_id': seq_id,
                    'videoTimeMs': target_times,
                    'role': role,
                    'player_id': player_id,
                    'supcon_label': group['supcon_label'].iloc[0],
                    # 'is_home_possession': group['is_home_possession'].iloc[0],
                    **interp_feats
                }))

        if dropped_agents:
            print(f"  -> {dropped_agents} agent-sequences had zero valid tracking points "
                  f"and were correctly excluded (not zero-filled).")

        self.final_extracted_df = pd.concat(resampled_rows, ignore_index=True)
        print(f"Interpolation complete. Every sequence now has exactly {self.sample_size} synchronized frames.")

    # PHASE 5: OUTPUT & COMPILATION
    def validate_extraction(self, sample_seq=10):
        """
        Validates the final extracted DataFrame to ensure metadata
        and tracking data merged correctly.
        """
        df = self.final_extracted_df

        print(f"\nPhase 5: Validation Check")
        print(f"Total Extracted Records: {len(df)}")

        # 1. Validate Columns
        expected_cols = ['seq_id', 'videoTimeMs', 'role', 'player_id', 'x', 'y', 'rel_x', 'rel_y', 'supcon_label',
                         'is_home_possession']
        missing_cols = [col for col in expected_cols if col not in df.columns]

        if missing_cols:
            print(f"WARNING: Missing expected columns: {missing_cols}")
        else:
            print("Successfully verified all required columns are present.")

            # 2. Check for missing metadata (ensures the Elastic Window join worked)
            missing_labels = df['supcon_label'].isna().sum()
            if missing_labels > 0:
                print(f"WARNING: Found {missing_labels} records missing a 'supcon_label'.")
            else:
                print("All records successfully mapped to a SupCon label.")

        # 3. Sequence Specific Validation
        print(f"\n--- Sequence {sample_seq} Validation ---")
        sample_records = df[df['seq_id'] == sample_seq]

        if len(sample_records) > 0:
            print(f"Total records (players + ball): {len(sample_records)}")

            # Group by video time to see how many actual tracking frames exist
            unique_frames = sample_records['videoTimeMs'].nunique()
            print(f"Total unique tracking frames: {unique_frames}")

            # Print the first row as a dictionary to visually verify the data types
            print(f"\nSample Data Row:")
            sample_dict = sample_records.iloc[1].to_dict()
            for key, value in sample_dict.items():
                print(f"  {key}: {value}")
        else:
            print(f"Sequence {sample_seq} not found in extracted data.")
            print("(Note: This is normal if Sequence 10 did not contain a target Shot, Cross, or Foul).")

        # NaN audit — surfaces exactly which columns still have unresolved gaps.
        numeric_cols = ['x', 'y', 'z', 'speed', 'dist_to_goal']
        nan_report = df[numeric_cols].isna().mean() * 100
        print("\n--- NaN Audit (% missing per column) ---")
        print(nan_report.to_string())

        print("-----------------------------\n")

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
        feature_cols = ['x', 'y', 'z', 'speed', 'visibility', 'is_attacking', 'dist_to_goal']
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
                ball = frame_df[frame_df['role'] == 2]
                if not ball.empty:
                    seq_tensor[t_idx, 0, :] = ball[feature_cols].values[0]

                # B. Home Players (Role 0) -> Always Indices 1 through 11
                home = frame_df[frame_df['role'] == 0].sort_values('player_id')
                num_home = min(len(home), 11)  # Hard cap at 11 to prevent tensor shape errors
                if num_home > 0:
                    seq_tensor[t_idx, 1:1 + num_home, :] = home[feature_cols].values[:num_home]

                # C. Away Players (Role 1) -> Always Indices 12 through 22
                away = frame_df[frame_df['role'] == 1].sort_values('player_id')
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

if __name__ == '__main__':
    game_ids = [
        # '10511', '3812', '3813', '3814',
        # '3815', '3816', '3817', '3818',
        # '3819', '3820', '3821', '3822',
        # '3823', '3824', '3825', '3826',
        # '3827', '3828', '3829', '3830',
        # '3831', '10502', '10503', '10504',
        # '10505', '10507', '10509','10510',
        # '10512', '10513', '10514', '10515',
        # '10516', '3834', '3835', '3836',
        # '3837', '3838', '3839', '3840',
        # '3841', '3842', '3843', '3844',
        # '3845', '3846', '3847', '3857',
        # '3858', '3859', '3848', '3849',
        # '3850', '3852', '3832', '3853',
        # '3854', '3855', '3856', '3851',
        # '3833', '10508', '10506','10517',


    ]

    failed_games = []
    for gid in game_ids:
        print(f"Processing Game {gid}...")
        try:
            FIFAWC22(
                '../FIFA World Cup 2022',
                gid, save_Tensor=True,save_folder="Processed Tensors v2")
        except Exception as e:
            print(f"  !! Game {gid} failed: {e}")
            failed_games.append((gid, str(e)))

    if failed_games:
        print("\n=== Games that failed extraction ===")
        for gid, err in failed_games:
            print(f"  {gid}: {err}")