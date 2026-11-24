#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write a caption.h5 from a fixed caption string, skipping the Qwen3-VL captioner.

Same as ArtiFixer's output/jobs/write_caption.py, and the same on-disk format as
data_processing.captioning.generate_captions.generate_caption_hdf5: dataset "frame_00000" holding the
UMT5 last_hidden_state (bf16 bits as uint16, [tokens, 4096]), attrs caption and image_indices=[0].
The object-removal "bighull" variant uses it for its empty-floor prompt.

  python -m artifixer_gs.caption --caption TEXT --output_path P.h5 --text_encoder_model_id <Wan2.1 dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def write_caption(caption: str, output_path: Path, text_encoder_model_id: str, max_sequence_length: int = 512) -> Path:
    import h5py
    import torch

    from data_processing.captioning.generate_captions import generate_text_embedding, get_text_encoder_and_tokenizer

    text_encoder, tokenizer = get_text_encoder_and_tokenizer(text_encoder_model_id)
    with torch.inference_mode():
        embedding = generate_text_embedding(caption, text_encoder, tokenizer, max_sequence_length)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as hf:
        dataset = hf.create_dataset("frame_00000", data=embedding)
        dataset.attrs["caption"] = caption
        dataset.attrs["image_indices"] = np.array([0])
    print(f"Wrote {output_path} embedding={embedding.shape}", flush=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--caption", required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--text_encoder_model_id", required=True, help="Wan2.1 Diffusers dir (text_encoder/, tokenizer/).")
    parser.add_argument("--max_sequence_length", type=int, default=512)
    args = parser.parse_args()
    write_caption(args.caption, args.output_path, args.text_encoder_model_id, args.max_sequence_length)


if __name__ == "__main__":
    main()
