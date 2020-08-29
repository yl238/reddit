import logging

from random_forest_model.config import config
from random_forest_model.config import logging_config


VERSION_PATH = config.PACKAGE_ROOT / "VERSION"
LOG_PATH = config.PACKAGE_ROOT /'model.log'

# configure logger for use in package
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging_config.get_file_handler(LOG_PATH))
logger.propagate = False

with open(VERSION_PATH, 'r') as version_file:
    __version__ = version_file.read().strip()