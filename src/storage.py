"""Publish and fetch the exported champion model through S3-compatible storage.

Config comes from the environment, so the same code runs against LocalStack
(set S3_ENDPOINT_URL) or real AWS (leave it unset):
    S3_BUCKET        bucket name (default: term-deposit-models)
    S3_ENDPOINT_URL  e.g. http://localhost:4566 for LocalStack

CLI:
    python -m src.storage upload   [--src models/champion] [--prefix champion]
    python -m src.storage download [--dest models/champion_from_s3] [--prefix champion]
"""

import argparse
import os
import shutil
from pathlib import Path

import boto3


def get_client():
    """endpoint_url=None means real AWS; a URL redirects every call (e.g. LocalStack)."""
    return boto3.client("s3", endpoint_url=os.getenv("S3_ENDPOINT_URL"))


def bucket_name() -> str:
    return os.getenv("S3_BUCKET", "term-deposit-models")


def upload_model(src: Path, prefix: str = "champion") -> int:
    """Upload every file under src to s3://<bucket>/<prefix>/... Returns the file count."""
    src = Path(src)
    if not (src / "MLmodel").exists():
        raise ValueError(f"{src} is not an exported model (no MLmodel file)")
    prefix = prefix.strip("/")
    client, bucket = get_client(), bucket_name()
    count = 0
    for path in sorted(src.rglob("*")):
        if path.is_file():
            key = f"{prefix}/{path.relative_to(src).as_posix()}"
            client.upload_file(str(path), bucket, key)
            count += 1
    return count


def download_model(dest: Path, prefix: str = "champion") -> int:
    """Download s3://<bucket>/<prefix>/... into dest (replaced fresh). Returns the file count."""
    dest = Path(dest).resolve()
    prefix = prefix.strip("/")
    client, bucket = get_client(), bucket_name()

    keys = []
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=f"{prefix}/"
    ):
        keys += [o["Key"] for o in page.get("Contents", []) if not o["Key"].endswith("/")]
    if not keys:
        raise FileNotFoundError(f"No objects under s3://{bucket}/{prefix}/")

    if dest.exists():
        shutil.rmtree(dest)  # always fetch fresh, never mix versions
    for key in keys:
        target = (dest / key[len(prefix) + 1 :]).resolve()
        if dest not in target.parents:  # never write outside dest, whatever the key says
            raise ValueError(f"Unsafe object key: {key}")
        target.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(target))

    if not (dest / "MLmodel").exists():
        raise ValueError(f"Download looks wrong: no MLmodel file in {dest}")
    return len(keys)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("upload")
    up.add_argument("--src", default="models/champion")
    up.add_argument("--prefix", default="champion")
    down = sub.add_parser("download")
    down.add_argument("--dest", default="models/champion_from_s3")
    down.add_argument("--prefix", default="champion")
    args = parser.parse_args(argv)

    if args.cmd == "upload":
        n = upload_model(Path(args.src), args.prefix)
        print(f"Uploaded {n} files -> s3://{bucket_name()}/{args.prefix}/")
    else:
        n = download_model(Path(args.dest), args.prefix)
        print(f"Downloaded {n} files -> {args.dest}")


if __name__ == "__main__":
    main()
