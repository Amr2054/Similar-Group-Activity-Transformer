"""Post-extraction sanity checks: column presence, label coverage, per-sequence spot check, NaN audit."""

class ValidationMixin:
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