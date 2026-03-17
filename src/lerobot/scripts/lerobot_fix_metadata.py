#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Repair missing depth-video episode metadata for previously recorded datasets.

Background
----------
Some datasets were recorded with depth videos archived as MKV files but without
the per-episode metadata columns:
- videos/<depth_key>/chunk_index
- videos/<depth_key>/file_index
- videos/<depth_key>/from_timestamp
- videos/<depth_key>/to_timestamp

Aggregation relies on these fields. This script reconstructs them by:
1. reading episode lengths and dataset fps,
2. scanning depth MKV files under videos/<depth_key>/chunk-*/file-*.mkv,
3. assigning episodes sequentially to files,
4. writing the reconstructed metadata back to meta/episodes parquet files.

Example
-------
python -m lerobot.scripts.lerobot_fix_metadata \
  --repo-id user/my_dataset \
  --root /path/to/local/cache/user/my_dataset
"""

import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import DEFAULT_EPISODES_PATH
from lerobot.datasets.video_utils import get_video_duration_in_s


def _parse_chunk_file_indices(path: Path) -> tuple[int, int]:
	chunk_idx = int(path.parent.name.split("-")[-1])
	file_idx = int(path.stem.split("-")[-1])
	return chunk_idx, file_idx


def _list_depth_mkv_files(dataset_root: Path, depth_key: str) -> list[tuple[int, int, Path]]:
	root = dataset_root / "videos" / depth_key
	files = sorted(root.glob("chunk-*/file-*.mkv"))
	indexed = [(*_parse_chunk_file_indices(path), path) for path in files]
	indexed.sort(key=lambda x: (x[0], x[1]))
	return indexed


def _reconstruct_depth_assignments(
	episode_indices: list[int],
	episode_lengths: list[int],
	fps: int,
	depth_files: list[tuple[int, int, Path]],
	tolerance_s: float = 0.25,
) -> dict[int, dict[str, float | int]]:
	if not depth_files:
		raise FileNotFoundError("No depth .mkv files found for this depth key.")

	file_infos = []
	for chunk_idx, file_idx, path in depth_files:
		duration = float(get_video_duration_in_s(path))
		file_infos.append((chunk_idx, file_idx, path, duration))

	assignments: dict[int, dict[str, float | int]] = {}
	file_ptr = 0
	offset = 0.0

	for ep_idx, ep_len in zip(episode_indices, episode_lengths, strict=True):
		ep_dur = float(ep_len) / float(fps)

		while file_ptr < len(file_infos):
			chunk_idx, file_idx, _, file_duration = file_infos[file_ptr]

			if offset + ep_dur <= file_duration + tolerance_s:
				from_ts = max(0.0, offset)
				to_ts = from_ts + ep_dur
				assignments[ep_idx] = {
					"chunk_index": chunk_idx,
					"file_index": file_idx,
					"from_timestamp": from_ts,
					"to_timestamp": to_ts,
				}
				offset = to_ts

				if offset >= file_duration - tolerance_s:
					file_ptr += 1
					offset = 0.0
				break

			file_ptr += 1
			offset = 0.0

		if ep_idx not in assignments:
			raise RuntimeError(
				"Unable to reconstruct depth metadata: not enough depth files or inconsistent durations. "
				f"Failed at episode_index={ep_idx}."
			)

	return assignments


def _write_reconstructed_metadata(
	dataset_root: Path,
	per_depth_assignments: dict[str, dict[int, dict[str, float | int]]],
	backup: bool,
	dry_run: bool,
) -> int:
	episodes_root = dataset_root / "meta" / "episodes"
	parquet_paths = sorted(episodes_root.glob("chunk-*/file-*.parquet"))
	if not parquet_paths:
		raise FileNotFoundError(f"No episode metadata parquet files found in {episodes_root}")

	updated_rows = 0
	for path in parquet_paths:
		df = pd.read_parquet(path)
		if "episode_index" not in df.columns:
			raise ValueError(f"Missing 'episode_index' column in {path}")

		changed = False
		ep_indices = df["episode_index"].astype(int).to_numpy()

		for depth_key, assignments in per_depth_assignments.items():
			chunk_col = f"videos/{depth_key}/chunk_index"
			file_col = f"videos/{depth_key}/file_index"
			from_col = f"videos/{depth_key}/from_timestamp"
			to_col = f"videos/{depth_key}/to_timestamp"

			if chunk_col not in df.columns:
				df[chunk_col] = pd.NA
			if file_col not in df.columns:
				df[file_col] = pd.NA
			if from_col not in df.columns:
				df[from_col] = pd.NA
			if to_col not in df.columns:
				df[to_col] = pd.NA

			for i, ep_idx in enumerate(ep_indices):
				if ep_idx not in assignments:
					continue
				rec = assignments[ep_idx]
				df.at[df.index[i], chunk_col] = int(rec["chunk_index"])
				df.at[df.index[i], file_col] = int(rec["file_index"])
				df.at[df.index[i], from_col] = float(rec["from_timestamp"])
				df.at[df.index[i], to_col] = float(rec["to_timestamp"])
				updated_rows += 1
				changed = True

		if changed and not dry_run:
			if backup:
				backup_path = path.with_suffix(path.suffix + ".bak")
				if not backup_path.exists():
					shutil.copy2(path, backup_path)
			df.to_parquet(path)

	return updated_rows


def fix_depth_metadata(
	repo_id: str,
	root: str | Path | None = None,
	backup: bool = True,
	dry_run: bool = False,
	tolerance_s: float = 0.25,
) -> None:
	meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)

	if len(meta.depth_keys) == 0:
		logging.info("No depth keys found in features. Nothing to repair.")
		return

	episodes_df = meta.episodes.to_pandas()[["episode_index", "length"]].sort_values("episode_index")
	episode_indices = episodes_df["episode_index"].astype(int).tolist()
	episode_lengths = episodes_df["length"].astype(int).tolist()

	per_depth_assignments: dict[str, dict[int, dict[str, float | int]]] = {}
	for depth_key in meta.depth_keys:
		depth_files = _list_depth_mkv_files(meta.root, depth_key)
		if not depth_files:
			logging.warning(f"Skipping depth key '{depth_key}': no MKV files found.")
			continue

		per_depth_assignments[depth_key] = _reconstruct_depth_assignments(
			episode_indices=episode_indices,
			episode_lengths=episode_lengths,
			fps=meta.fps,
			depth_files=depth_files,
			tolerance_s=tolerance_s,
		)

	if not per_depth_assignments:
		logging.info("No depth metadata updates were generated.")
		return

	updated_rows = _write_reconstructed_metadata(
		dataset_root=meta.root,
		per_depth_assignments=per_depth_assignments,
		backup=backup,
		dry_run=dry_run,
	)

	if dry_run:
		logging.info(
			f"Dry run complete. Would update {updated_rows} rows in {DEFAULT_EPISODES_PATH} shard files."
		)
	else:
		logging.info(
			f"Depth metadata repair complete. Updated {updated_rows} rows in {DEFAULT_EPISODES_PATH} shard files."
		)


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Repair missing depth metadata in LeRobot episode parquet files")
	parser.add_argument("--repo-id", type=str, required=True, help="Dataset repository id (e.g. user/dataset)")
	parser.add_argument(
		"--root",
		type=str,
		default=None,
		help="Optional local dataset root path. Defaults to ~/.cache/huggingface/lerobot/<repo-id>",
	)
	parser.add_argument(
		"--no-backup",
		action="store_true",
		help="Disable creating .bak copies of episode parquet files before writing",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Compute and validate mappings, but do not write any files",
	)
	parser.add_argument(
		"--tolerance-s",
		type=float,
		default=0.25,
		help="Duration tolerance (seconds) when fitting episodes into depth files",
	)
	return parser


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
	args = _build_parser().parse_args()
	fix_depth_metadata(
		repo_id=args.repo_id,
		root=args.root,
		backup=not args.no_backup,
		dry_run=args.dry_run,
		tolerance_s=args.tolerance_s,
	)


if __name__ == "__main__":
	main()
