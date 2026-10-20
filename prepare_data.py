"""Download Fox (or another character's) ranked replays and convert them to raw frames.

Processes one archive at a time: download it, parse every replay on all cores, write
one .npz per kept game and a line per game to index.jsonl, then delete the archive.
Rerunning skips finished archives and games already in the index.

Usage:
    python prepare_data.py                                   # Fox, master-master + master-diamond, all 6 archives
    python prepare_data.py --ranks master-master --archives 1 2
    python prepare_data.py --character LINK --ranks master-diamond --archives 3 --data-dir D:\\smash_smoke
"""

import argparse
import collections
import concurrent.futures as cf
import json
import os
import tarfile
import time

import melee
from huggingface_hub import hf_hub_download

from smash_rl.replays import REPO, convert_safely, read_index, rewrite_index


def parse_args():
    p = argparse.ArgumentParser(description="Download and parse ranked Melee replays")
    p.add_argument("--data-dir", default=r"D:\smash_data", help="where archives, parsed games and index.jsonl go")
    p.add_argument("--character", default="FOX", help="dataset folder and libmelee character name, e.g. FOX, FALCO")
    p.add_argument("--ranks", nargs="+", default=["master-master", "master-diamond"], help="rank pairs to download")
    p.add_argument("--archives", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6], help="source archives a1..a6")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2), help="parsing processes")
    p.add_argument("--keep-archives", action="store_true", help="don't delete archives after parsing")
    p.add_argument("--max-games", type=int, default=None, help="stop each archive after this many replays (testing)")
    return p.parse_args()


def process_archive(tar_path, name, rank, args, character, index_file, known):
    out_dir = os.path.join(args.data_dir, "parsed", name)
    os.makedirs(out_dir, exist_ok=True)
    counts, pending = collections.Counter(), {}

    def collect(done):
        for future in done:
            game_id = pending.pop(future)
            result = future.result()
            if "skip" in result:
                counts[result["skip"]] += 1
                continue
            entry = dict(result, archive=name, rank=rank, path=f"parsed/{name}/{game_id}.npz")
            index_file.write(json.dumps(entry) + "\n")
            index_file.flush()
            known.add(game_id)
            counts["kept"] += 1

    with cf.ProcessPoolExecutor(args.workers) as pool, tarfile.open(tar_path, "r:gz") as tar:
        seen = 0
        for member in tar:
            if not member.isfile() or not member.name.endswith(".slp"):
                continue
            if args.max_games is not None and seen >= args.max_games:
                break
            seen += 1
            game_id = os.path.splitext(os.path.basename(member.name))[0]
            if game_id in known:
                counts["already parsed"] += 1
                continue
            data = tar.extractfile(member).read()
            out_path = os.path.join(out_dir, game_id + ".npz")
            pending[pool.submit(convert_safely, data, game_id, out_path, character)] = game_id
            if len(pending) >= 4 * args.workers:
                done, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                collect(done)
        collect(cf.wait(pending)[0])
    return counts


def main():
    args = parse_args()
    character = melee.Character[args.character.upper()].value
    for sub in ("raw", "parsed", "done"):
        os.makedirs(os.path.join(args.data_dir, sub), exist_ok=True)
    index_path = os.path.join(args.data_dir, "index.jsonl")
    entries = read_index(index_path)
    known = {e["game_id"] for e in entries}
    rewrite_index(index_path, entries)  # drops a half-written last line from an interrupted run
    with open(index_path, "a") as index_file:
        for rank in args.ranks:
            for archive in args.archives:
                name = f"{args.character.upper()}_{rank}_a{archive}"
                marker = os.path.join(args.data_dir, "done", name)
                if os.path.exists(marker):
                    print(f"{name}: done already")
                    continue
                t = time.time()
                print(f"{name}: downloading...", flush=True)
                tar_path = hf_hub_download(REPO, f"{args.character.upper()}/{name}.tar.gz", repo_type="dataset",
                                           local_dir=os.path.join(args.data_dir, "raw"))
                print(f"{name}: parsing with {args.workers} workers...", flush=True)
                counts = process_archive(tar_path, name, rank, args, character, index_file, known)
                print(f"{name}: {dict(counts)} in {(time.time() - t) / 60:.1f} min", flush=True)
                if args.max_games is None:
                    open(marker, "w").close()
                    if not args.keep_archives:
                        os.remove(tar_path)
    print(f"index: {len(read_index(index_path))} games in {index_path}")


if __name__ == "__main__":
    main()
