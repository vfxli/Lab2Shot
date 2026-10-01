"""What the released Mesh4D model works on: plain data shared by nodes.py (the fewest frames it takes) and worker.py
(how it cuts a shot into windows), so the two cannot drift apart."""

from __future__ import annotations

WINDOW = 6  # frames the model sees at once (configs/OBJVERSE/train/infer.yaml length_sequence / num_frames): not a setting
