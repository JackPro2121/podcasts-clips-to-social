#!/usr/bin/env python3
"""Compatibility shim for the V1 podcast entrypoint.

The implementation now lives in ``v1/main.py``; this shim keeps existing
workflows, docs and tooling working unchanged.
"""

from v1.main import main

if __name__ == "__main__":
    main()
