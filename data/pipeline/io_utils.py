"""Raw JSON/bz2 loading for event and tracking data, plus jersey<->playerId mapping."""
import bz2
import json
import pandas as pd


class IOMixin:
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
