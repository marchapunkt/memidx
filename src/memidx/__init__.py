"""memidx: section-level hybrid search over the Markdown files of a git repository."""
__version__ = "0.2.0"

from .config import Config, load_config  # noqa: F401
from .core import configure, get_config, open_db, related, render_list, render_text, search, status, update  # noqa: F401
