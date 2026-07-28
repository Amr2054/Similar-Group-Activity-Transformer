"""FIFAWC22: orchestrates the full raw-JSON -> saved-tensor pipeline for one match."""
from .io_utils import IOMixin
from .event_engineering import EventEngineeringMixin
from .feature_extraction import FeatureExtractionMixin
from .resampling import ResamplingMixin
from .validation import ValidationMixin
from .tensor_export import TensorExportMixin
from .constants import DEFAULT_SAMPLE_SIZE, DEFAULT_PRE_BUFFER, DEFAULT_POST_BUFFER


class FIFAWC22(
    IOMixin, EventEngineeringMixin, FeatureExtractionMixin,
    ResamplingMixin, ValidationMixin, TensorExportMixin,
):
    """
    End-to-end data engineering pipeline for PFF football tracking data.
    Ingests raw PFF Event and Tracking JSONs, extracts tactical sequences
    (shots, crosses), normalizes pitch coordinates, and compiles the result
    into [sample_size, 23, len(FEATURE_COLS)] tensors for SupCon training.
    """

    def __init__(self, folder_path, game_id,
                 sample_size=DEFAULT_SAMPLE_SIZE,
                 pre_buffer=DEFAULT_PRE_BUFFER,
                 post_buffer=DEFAULT_POST_BUFFER,
                 save_Tensor=False,
                 save_folder="Processed Tensors"):
        self.folder_path = folder_path
        self.game_id = game_id
        self.sample_size = sample_size
        self.pre_buffer = pre_buffer
        self.post_buffer = post_buffer
        self.save_folder = save_folder

        self.load_event_data()
        self.get_important_sequences()
        self.load_tracking_data()
        self.extract_per_frame_info()
        self.post_process_ball_data()
        self.interpolate_to_fixed_length()
        self.validate_extraction(sample_seq=10)

        if save_Tensor:
            self.save_to_tensor()