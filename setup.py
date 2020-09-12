import io
import os
import re
from pathlib import Path

from setuptools import find_packages
from setuptools import setup

# Package metadata
NAME = 'reddit'
DESCRIPTION = "A linear SVC model to classify Reddit posts"
URL = "https://github.com/yusueliu/reddit"
EMAIL = "sue.liu@gmail.com"
AUTHOR = "Sue Liu"
REQUIRES_PYTHON = '>=3.6.0'

# Packages that are required for this module to be executed
def list_reqs(fname='requirements.txt'):
    with open(fname) as f:
        return f.read().splitlines()

def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


# Load the package's __version__.py module as a dictionary
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / 'reddit'
README = (PACKAGE_DIR / "README.md").read_text()

about = {}
with open(PACKAGE_DIR / 'VERSION') as f:
    _version = f.read().strip()
    about['__version__'] = _version

setup(
    name=NAME,
    version=about['__version__'],
    url=URL,
    license='MIT',
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=README,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=('tests', 'notebooks',)),
    package_data={NAME: ['VERSION']},
    install_requires=list_reqs(),
    extra_require = {},
    include_package_data=True,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
