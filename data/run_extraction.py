"""Batch driver: runs the FIFAWC22 pipeline over every configured match and reports failures without aborting the run."""
from data.pipeline import FIFAWC22

GAME_IDS = [
    '10502', '10503', '10504', '10505', '10506', '10507', '10508', '10509',
    # '10510', '10511', '10512', '10513', '10514', '10515', '10517',
    # '3812', '3814', '3815', '3818', '3819', '3820', '3821', '3822', '3823',
    # '3824', '3825', '3828', '3829', '3830', '3831', '3832', '3833', '3834',
    # '3835', '3836', '3837', '3839', '3840', '3841', '3842', '3844',
]
DATA_ROOT = '../FIFA World Cup 2022'


def main():
    failed_games = []
    for gid in GAME_IDS:
        print(f"Processing Game {gid}...")
        try:
            FIFAWC22(DATA_ROOT, gid, save_Tensor=True,save_folder="Processed Tensors v3")
        except Exception as e:
            print(f"  !! Game {gid} failed: {e}")
            failed_games.append((gid, str(e)))

    if failed_games:
        print("\n=== Games that failed extraction ===")
        for gid, err in failed_games:
            print(f"  {gid}: {err}")


if __name__ == '__main__':
    main()