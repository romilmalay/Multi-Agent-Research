"""Multi-agent research pipeline. Importable without FastAPI or a database.

Library discipline: importing this package must never change the host
application's logging. Only an entrypoint (`cli.py`, `app/main.py`,
`app/worker.py`) may call `logging.configure_logging()`. The NullHandler below
keeps an unconfigured import silent instead of falling back to stderr.
"""

import logging

__version__ = "0.1.0"

logging.getLogger(__name__).addHandler(logging.NullHandler())
