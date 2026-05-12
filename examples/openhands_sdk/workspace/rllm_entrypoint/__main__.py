"""Entry point for `python3 -m rllm_entrypoint`."""
import logging
import sys

from rllm_entrypoint.runner import run

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    sys.exit(run())
