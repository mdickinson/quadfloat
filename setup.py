import os.path

from setuptools import setup, find_packages


LONG_DESCRIPTION = """\

The quadfloat package is intended to be a pure Python unofficial reference
implementation of the IEEE 754 binary interchange formats, providing almost
all of the recommended operations described in the standard.
"""

PROJECT_URL = "https://github.com/mdickinson/quadfloat"


def get_version_info():
    """Extract version information as a dictionary from version.py."""
    version_info = {}
    with open(os.path.join("quadfloat", "version.py"), 'r') as f:
        version_code = compile(f.read(), "version.py", 'exec')
        exec(version_code, version_info)
    return version_info


version_info = get_version_info()

setup(
    name='quadfloat',
    version=version_info['release'],
    author='Mark Dickinson',
    author_email="dickinsm@gmail.com",
    url=PROJECT_URL,
    description='IEEE 754 binary floating-point',
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
    ],
)
