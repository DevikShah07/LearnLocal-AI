"""core/logger.py"""
import logging
import sys
from core.config import settings

import logging
import sys
from core.config import settings

# Create logger
logger = logging.getLogger("LearnLocal")
logger.setLevel(getattr(logging, settings.LOG_LEVEL, logging.INFO))

# Create formatters
formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)

# File handler
fh = logging.FileHandler("app.log", encoding="utf-8")
fh.setFormatter(formatter)
logger.addHandler(fh)
