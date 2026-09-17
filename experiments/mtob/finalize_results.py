#!/usr/bin/env python3
"""Offline MTOB audit/score/analyze entrypoint. No generation or model loading."""
import argparse
import os
import socket
import sys
from pathlib import Path


def deny_network(*args, **kwargs):
    raise RuntimeError("Network access is forbidden in MTOB evaluation_v1")


def install_write_guard():
    root = (Path(__file__).resolve().parent / "evaluation_v1").resolve()
    (root / "tmp").mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(root / "tmp")
    def check(path):
        if isinstance(path, (str, bytes, os.PathLike)):
            resolved = Path(os.fsdecode(path)).resolve()
            if resolved != root and root not in resolved.parents:
                raise PermissionError(f"Write outside evaluation_v1 rejected: {resolved}")
    def guard(event, args):
        if event == "open":
            path, mode, flags = args
            if (isinstance(mode, str) and any(c in mode for c in "wax+")) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                check(path)
        elif event in {"os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.truncate"}:
            check(args[0])
        elif event in {"os.rename", "os.link", "os.symlink"}:
            check(args[0]); check(args[1])
        elif event.startswith("socket.") and event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
            deny_network()
    sys.dont_write_bytecode = True
    sys.addaudithook(guard)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "score", "analyze", "all"), required=True)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    socket.create_connection = deny_network
    socket.getaddrinfo = deny_network
    install_write_guard()
    from mtob_grammar.closure_audit import OUT, audit, save
    from mtob_grammar.closure_evaluation import analyze, score, write_report
    save(OUT / f"{args.stage}_status.json", {"state": "running"})
    try:
        if args.stage in ("audit", "all"):
            report = audit()
            write_report()
            if not report["passed"]:
                raise RuntimeError("Audit failed; generation is not authorized to repair inconsistencies")
        if args.stage in ("score", "all"):
            score()
        if args.stage in ("analyze", "all"):
            analyze()
        save(OUT / f"{args.stage}_status.json", {"state": "complete"})
    except Exception as exc:
        save(OUT / f"{args.stage}_status.json", {"state": "error", "error": f"{type(exc).__name__}: {exc}"})
        save(OUT / f"{args.stage}_error.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
