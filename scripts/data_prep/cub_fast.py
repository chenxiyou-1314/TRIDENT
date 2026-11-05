#!/usr/bin/env python3
"""Robust multithreaded downloader for CUB_200_2011.tgz from Caltech S3.

The Caltech record 302-redirects to an S3 presigned URL that EXPIRES in ~60s,
and the endpoint rate-limits each connection to ~30 KB/s. A single large chunk
therefore can never finish within the 60 s window, so we split the file into
small 1 MB chunks, re-resolve a FRESH presigned URL right before every chunk
GET, and stream with bounded retries. Aggregate throughput scales with the
number of parallel connections (per-connection limit, not per-IP).
"""
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
OUT = "/hy-tmp/TRIDENT/datasets/CUB_200_2011.tgz"
TMP = OUT + ".parts"
CHUNK = 1 << 20          # 1 MB chunks (finish within 60s @ ~30KB/s)
CONN = 24                # parallel connections
RETRIES = 8
TIMEOUT = 120

# S3/CloudFront rejects urllib's default UA (403); send a browser-like UA.
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}


def resolve():
    req = urllib.request.Request(
        BASE, headers={**UA, "Range": "bytes=0-0"})
    for _ in range(RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                size = int(r.headers["Content-Range"].split("/")[-1])
                return r.geturl(), size
        except Exception:
            time.sleep(2)
    raise RuntimeError("resolve() failed after retries")


def fetch_chunk(idx, start, end):
    expected = end - start + 1
    last = None
    for attempt in range(RETRIES):
        try:
            url, _ = resolve()
            req = urllib.request.Request(
                url, headers={**UA, "Range": "bytes=%d-%d" % (start, end)})
            data = b""
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                while True:
                    buf = r.read(1 << 16)
                    if not buf:
                        break
                    data += buf
            if len(data) == expected:
                with open(os.path.join(TMP, "c%07d" % idx), "wb") as f:
                    f.write(data)
                return idx, len(data)
            last = "short %d/%d" % (len(data), expected)
        except Exception as e:
            last = repr(e)
        time.sleep(3 + attempt)
    return idx, -1, last


def main():
    os.makedirs(TMP, exist_ok=True)
    url, size = resolve()
    n = (size + CHUNK - 1) // CHUNK
    print("size=%d chunks=%d" % (size, n))
    sys.stdout.flush()

    done = set()
    for fn in os.listdir(TMP):
        if fn.startswith("c") and fn[1:].isdigit():
            done.add(int(fn[1:]))

    tasks = []
    for i in range(n):
        if i in done:
            continue
        start = i * CHUNK
        end = min((i + 1) * CHUNK, size) - 1
        tasks.append((i, start, end))
    print("resume: %d/%d chunks already done" % (len(done), n))
    sys.stdout.flush()

    ok = len(done)
    with ThreadPoolExecutor(max_workers=CONN) as ex:
        futs = [ex.submit(fetch_chunk, i, s, e) for (i, s, e) in tasks]
        for fut in as_completed(futs):
            res = fut.result()
            if len(res) == 2 and res[1] > 0:
                ok += 1
                if ok % 25 == 0 or ok == n:
                    print("[ok] %d/%d" % (ok, n))
                    sys.stdout.flush()
            else:
                print("[FAIL] chunk", res)
                sys.stdout.flush()
                ex.shutdown(wait=False, cancel_futures=True)
                raise RuntimeError("chunk failed: %s" % str(res))

    print("concatenating...")
    sys.stdout.flush()
    with open(OUT, "wb") as out:
        for i in range(n):
            p = os.path.join(TMP, "c%07d" % i)
            with open(p, "rb") as f:
                out.write(f.read())
    print("DONE size=%d" % os.path.getsize(OUT))


if __name__ == "__main__":
    main()
