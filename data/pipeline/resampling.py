"""Resamples every (seq_id, role, player_id) trajectory onto a fixed, evenly-spaced time grid via linear interpolation."""
import numpy as np
import pandas as pd
from .constants import FEATURE_COLS


class ResamplingMixin:
    def interpolate_to_fixed_length(self):
        """
        Resamples every (seq_id, role, player_id) trajectory onto a shared
        grid of exactly `sample_size` evenly-spaced timestamps per sequence,
        via linear interpolation on real elapsed time.
        """
        print("Phase 4b: Interpolating all sequences onto a fixed time grid")
        interp_cols = FEATURE_COLS
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
