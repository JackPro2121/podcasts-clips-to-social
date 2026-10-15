#!/usr/bin/env python3
"""Compatibility shim for the V2 streamer entrypoint.

The implementation now lives in ``v2/main_streamer.py``.
"""

from v2.main_streamer import main

if __name__ == "__main__":
    main()
