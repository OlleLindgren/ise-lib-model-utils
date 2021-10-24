import datetime
import configparser
from pathlib import Path

# ########## COMMON ISE SETUP ################

# For lib packages, __file__ should be in ISE_ROOT/lib/ise-this-package/src
# This is different to the normal ISE_ROOT/ise-this-package/src
ISE_ROOT: Path = Path(__file__).parents[3]

COMMON_CONFIG: Path = ISE_ROOT/'ise-config.cfg'
config = configparser.ConfigParser()
config.read(COMMON_CONFIG)
PATHS = config['PATHS']

DATA_ROOT: Path = ISE_ROOT/PATHS['data_root']
CACHE_ROOT: Path = ISE_ROOT/PATHS['cache_root'] / 'inference'

# ########## PACKAGE-SPECIFIC ################

DATA_HIERARCHY: list = ['period', 'dataset_key', 'ticker']
SPS_KEY: str = 'sps'

# Model stuff

TRAIN_TEST_RATIO: float = 0.85
MAX_XNA_RATIO: float = 0.5

# Technical settings
MAX_CACHE_AGE: datetime.timedelta = datetime.timedelta(days=7)

DATA_ROOT, DATA_HIERARCHY, SPS_KEY, TRAIN_TEST_RATIO, MAX_XNA_RATIO, CACHE_ROOT, MAX_CACHE_AGE