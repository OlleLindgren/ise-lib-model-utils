import setuptools
from pathlib import Path

SRC_ROOT = Path(__file__).parent

with open(SRC_ROOT / "README.md", "r") as fh:
    long_description = fh.read()

with open(SRC_ROOT / 'requirements.txt', "r") as f:
    requirements = [r.strip() for r in f.readlines() if r.strip()
        if not r.startswith('git+')]
    git_requirements = [r.strip() for r in f.readlines() if r.strip()
        if r.startswith('git+')]

# If any git requirements, install them separately
if git_requirements:
    import sys
    import subprocess
    for requirement in git_requirements:
        subprocess.run([sys.executable, '-m', 'pip', 'install', requirement])

with open(SRC_ROOT / '__init__.py', "r") as f:
    __VERSION_LINE = next(filter(lambda s: 'version' in s, f.readlines()))
    VERSION = __VERSION_LINE.split('=')[-1].strip(" \n\"")

setuptools.setup(
    name="ise-lib-model-utils",
    version=VERSION,
    author="Olle Lindgren",
    author_email="lindgrenolle@live.se",
    description="ISE model I/O package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OlleLindgren/ise-lib-model-utils",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=["Programming Language :: Python :: 3",],
    python_requires='>=3.8',
)
