"""Tactical anchor extraction, compound SupCon label construction, and elastic tracking-window calculation."""
from collections import Counter
import pandas as pd
from .constants import TARGET_EVENT_TYPES


class EventEngineeringMixin:
    # PHASE 2: EVENT DATA ENGINEERING (LABELS & WINDOWS)
    def get_important_sequences(self):
        """
        Identifies tactical anchors, chains compound labels (e.g., CR_I__SH_S),
        and calculates precise elastic tracking windows clamped to sequence boundaries.
        """
        target_event_types = TARGET_EVENT_TYPES
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
