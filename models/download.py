#!/usr/bin/env python3
"""
Download the tiny model + tokenizer used to bring up the basicllm Llama 2 runner.

These are Andrej Karpathy's llama2.c artifacts:
  * stories15M.bin - a real (small) Llama-2-architecture model, fp32, trained on
                     the TinyStories dataset. Tiny enough to load and run on CPU
                     in seconds, which makes it ideal for validating the runner.
  * tokenizer.bin  - the Llama 2 SentencePiece tokenizer (32000 tokens) in
                     llama2.c's binary format.

Files are saved next to this script (the models/ folder). Uses only the Python
standard library, so no `pip install` is needed.

Usage:
    python download.py            # download any missing files
    python download.py --force    # re-download everything
"""

import os
import sys
import struct
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# filename -> (source URL, what it is, approximate size)
FILES = {
    "stories15M.bin": (
        "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin",
        "15M-param Llama-2 model (fp32), trained on TinyStories by Andrej Karpathy",
        "~58 MB",
    ),
    "tokenizer.bin": (
        "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin",
        "Llama 2 SentencePiece tokenizer (32000 tokens), llama2.c binary format",
        "~424 KB",
    ),
}


def _progress(block_num, block_size, total_size):
    if total_size <= 0:
        return
    done = min(total_size, block_num * block_size)
    pct = done * 100 // total_size
    mb = done / (1024 * 1024)
    sys.stdout.write(f"\r         {pct:3d}%  ({mb:6.1f} MB)")
    sys.stdout.flush()


def download(name, url, force=False):
    dest = os.path.join(HERE, name)
    if os.path.exists(dest) and not force:
        print(f"  [skip] {name} already present ({os.path.getsize(dest):,} bytes)")
        return
    print(f"  [get ] {name}")
    print(f"         from {url}")
    urllib.request.urlretrieve(url, dest, _progress)
    print(f"\n         saved {os.path.getsize(dest):,} bytes -> {dest}")


def verify():
    """Sanity-check the downloads by reading the model header + tokenizer header."""
    mpath = os.path.join(HERE, "stories15M.bin")
    tpath = os.path.join(HERE, "tokenizer.bin")
    if os.path.exists(mpath):
        with open(mpath, "rb") as f:
            dim, hidden, n_layers, n_heads, n_kv, vocab, seq = struct.unpack("<7i", f.read(28))
        print(f"\n  stories15M.bin: dim={dim} hidden={hidden} layers={n_layers} "
              f"heads={n_heads} kv_heads={n_kv} vocab={abs(vocab)} seq_len={seq}")
    if os.path.exists(tpath):
        with open(tpath, "rb") as f:
            (max_tok_len,) = struct.unpack("<i", f.read(4))
        print(f"  tokenizer.bin:  max_token_length={max_tok_len}")


def main():
    force = "--force" in sys.argv
    print(f"Downloading basicllm model files into: {HERE}\n")
    for name, (url, desc, size) in FILES.items():
        print(f"{name}  ({size}) - {desc}")
        download(name, url, force)
        print()
    verify()
    print("\nDone.")


if __name__ == "__main__":
    main()
