"""PhysDrive dataset adapter stub."""

from pathlib import Path

from ir_stress.dataset.base import SessionMeta


class PhysDriveAdapter:
    """
    Placeholder adapter for the PhysDrive dataset.

    Expected layout (to be implemented):
      physdrive_root/
        <session>/
          ir_frames/     # IR video frames
          signals/       # multiple physiological signals (ECG, PPG, etc.)
    """

    def discover_sessions(self, raw_root: Path) -> list[SessionMeta]:
        raise NotImplementedError(
            "PhysDriveAdapter is not yet implemented. "
            "Subclass DatasetAdapter with PhysDrive-specific discovery logic."
        )

    def train_test_split(
        self, sessions: list[SessionMeta], val_subjects: list[int]
    ) -> tuple[list[SessionMeta], list[SessionMeta]]:
        raise NotImplementedError("PhysDriveAdapter.train_test_split is not yet implemented.")

    def preprocess_session(self, session: SessionMeta, out_dir: Path) -> Path:
        raise NotImplementedError("PhysDriveAdapter.preprocess_session is not yet implemented.")
