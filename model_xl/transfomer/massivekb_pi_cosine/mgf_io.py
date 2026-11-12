#!/usr/bin/env python3
"""Shared streaming MGF helpers.

Reads LF or CRLF input, normalizes line endings and rejects truncated blocks.
Peak/header text is otherwise preserved. Group filenames match the existing
collect_original_with_prediction convention. External xz failures are checked
by finish_stream; debug prefixes may deliberately close decompression early.
"""
from __future__ import annotations

import codecs
import hashlib
import lzma
import os
import re
import subprocess
import sys
from pathlib import Path

CHUNK = 1 << 26       # 64 MiB, for plain-file reads (measured 142 MB/s)
PIPE_CHUNK = 1 << 20  # 1 MiB, for subprocess-pipe reads: on Windows a 64 MiB pipe read
                      # collapses to ~32 MB/s while 256 KiB-4 MiB sustain ~94 MB/s
SEP = "\nEND IONS\n"
END_LINE = "\nEND IONS"


def open_binary_stream(path, threads: int = 0):
    """Return ``(binary_stream, subprocess_or_None, read_size)`` for a (possibly .xz) MGF.

    Read **bytes**: ``Popen(..., text=True)`` degenerates to ~0.35 MB/s on this platform.
    For .xz files the file is 1 stream / 644 blocks, so ``xz -T N`` decodes blocks in
    parallel (block order, and therefore bytes, are preserved): 108 MB/s at -T8 and
    153 MB/s at -T16, versus ~28 MB/s for single-threaded liblzma.
    """
    if threads == 0:
        threads = min(max(1, (os.cpu_count() or 2) - 1), 8)
    if threads > 1 and str(path).endswith(".xz"):
        try:
            proc = subprocess.Popen(["xz", "-dc", f"-T{int(threads)}", path],
                                    stdout=subprocess.PIPE)
            return proc.stdout, proc, PIPE_CHUNK
        except (OSError, ValueError) as exc:  # xz missing or no -T support
            print(f"  WARNING: xz -T{threads} unavailable ({exc}); falling back to lzma",
                  file=sys.stderr, flush=True)
    if str(path).endswith(".xz"):
        return lzma.open(path, "rb"), None, CHUNK
    return open(path, "rb"), None, CHUNK


def iter_blocks(stream, max_spectra=None, progress=None, label="", read_size: int = CHUNK):
    """Yield MGF block bodies from a binary stream (see the module docstring).

    `max_spectra` caps the stream (debug), `progress`/`label` print a stderr counter
    every N blocks.
    """
    dec = codecs.getincrementaldecoder("utf-8")("strict")
    buf = ""
    pending_cr = False  # a trailing "\\r" held back so CRLF never straddles a chunk
    n = 0
    while True:
        raw = stream.read(read_size)
        if not raw:
            text = dec.decode(b"", True)
            if pending_cr:
                text = "\r" + text
            buf += text
            break
        text = dec.decode(raw)
        if pending_cr:
            text = "\r" + text
            pending_cr = False
        if text.endswith("\r"):
            text = text[:-1]
            pending_cr = True
        buf += text.replace("\r\n", "\n")  # massKB's train file is plain LF; stay robust
        parts = buf.split(SEP)
        buf = parts.pop()  # incomplete tail stays buffered
        for blk in parts:
            n += 1
            yield blk
            if progress and n % progress == 0:
                print(f"  [{label}] {n:,} spectra...", file=sys.stderr, flush=True)
            if max_spectra and n >= max_spectra:
                print(f"  [{label}] {n:,} spectra total (capped by --max-spectra)",
                      file=sys.stderr, flush=True)
                return
    if buf.strip():
        tail = buf[:-len(END_LINE)] if buf.endswith(END_LINE) else buf
        if buf.endswith(END_LINE):
            n += 1
            yield tail
        else:
            raise ValueError("Incomplete MGF block at EOF (missing END IONS)")
    if progress:
        print(f"  [{label}] {n:,} spectra total", file=sys.stderr, flush=True)


def parse_block_key(block: str):
    """``(SEQ, CHARGE)`` of one MGF block, or None if either is missing.

    Plain string ops only (two ``find``s + slicing): blocks are ~4.5 kB and there are
    tens of millions of them, so anything per-block is hot.
    """
    block = "\n" + block
    i = block.find("\nSEQ=")
    if i < 0:
        return None
    i += 1
    j = block.find("\n", i)
    seq = block[i + 4: j if j >= 0 else len(block)].strip()
    i = block.find("\nCHARGE=")
    if i < 0 or not seq:
        return None
    i += 1
    j = block.find("\n", i)
    raw = block[i + 7: j if j >= 0 else len(block)].strip().rstrip("+").strip()
    if not raw.isdigit() or int(raw) < 1:
        return None
    return seq, int(raw)


def finish_stream(proc, capped=False):
    if proc:
        code = proc.wait()
        if code and not capped:
            raise RuntimeError(f"xz decompression failed with exit code {code}")


def headers_of(block):
    return dict(line.split("=", 1) for line in block.splitlines()
                if "=" in line and not line[:1].isdigit())


# --- filename convention: MUST match collect_original_with_prediction.make_group_filename ---
def sanitize_filename(name: str, max_len: int = 80) -> str:
    name = name.strip()
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:max_len] if name else "EMPTY_SEQ"


def make_group_filename(seq: str, charge: int) -> str:
    return f"{sanitize_filename(seq, 80)}_charge{charge}_{hashlib.md5(seq.encode('utf-8')).hexdigest()[:12]}.mgf"


def tag_block(block: str, source_tag: str) -> str:
    """Return the block with exactly one ``SOURCE=<source_tag>`` line, right after BEGIN IONS.

    Same convention as ``collect_original_with_prediction.replace_or_insert_source``.
    """
    out, inserted = [], False
    for line in block.split("\n"):
        s = line.strip()
        if s.upper().startswith("SOURCE="):
            if not inserted:
                out.append(f"SOURCE={source_tag}")
                inserted = True
            continue
        out.append(line)
        if s == "BEGIN IONS" and not inserted:
            out.append(f"SOURCE={source_tag}")
            inserted = True
    if not inserted:
        return f"BEGIN IONS\nSOURCE={source_tag}\n" + "\n".join(out)
    return "\n".join(out)


def write_block(path: Path, block: str, mode: str = "a") -> None:
    """Append one block to a group file, re-terminating it with ``END IONS``.

    Callers pass the block *body* (the text `iter_blocks` yields), which is everything
    from BEGIN IONS up to — but excluding — the ``END IONS`` terminator.
    """
    with open(path, mode, encoding="utf-8", newline="\n") as fh:
        fh.write(block.strip("\n") + "\nEND IONS\n")
