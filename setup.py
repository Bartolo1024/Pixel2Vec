import os
import re
import sys

from setuptools import find_packages, setup

is_for_windows = len(sys.argv) >= 3 and sys.argv[2].startswith("--plat-name=win")

if is_for_windows:
    scripts = None
    entry_points = {
        "console_scripts": [
            "imagine=imaginairy.cli.main:imagine_cmd",
            "aimg=imaginairy.cli.main:aimg",
        ],
    }
else:
    scripts = ["imaginairy/bin/aimg", "imaginairy/bin/imagine"]
    entry_points = None


def version():
    this_directory = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(this_directory, 'Pixel2Vec/version.py')) as f:
        version_file = f.read()
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
        version = version_match.group(1)

    return version


def readme():
    with open('Readme.md', encoding='utf-8') as f:
        return f.read()


setup(
    name="Pixel2Vec",
    version=version(),
    description="A self-supervised feature extraction trained on the single image.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    project_urls={
        "Documentation": "https://github.com/bartolo1024/Pixel2Vec/blob/master/Readme.md",
        "Source": "https://github.com/bartolo1024/Pixel2Vec",
    },
    url='https://github.com/bartolo1024/Pixel2Vec',
    keywords=['pytorch', 'computer-vision', 'unsupervised', 'deep-learning'],
    classifiers=[
        'Framework :: PyTorch',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    packages=find_packages(include=("pixel2vec", "pixel2vec.*")),
    package_data={"pixel2vec": ["configs/*.yaml", "data/*", "experiments/*.yaml"]},
    install_requires=[
        "click",
        "click-help-colors",
        "click-shell",
        "h5py",
        "pyyaml",
        "scikit-learn",
        "pandas",
        "numpy",
        "torch<=2.0.0",
        "torchvision>=0.13.1",
        "tqdm",
        "pytorch-ignite",
        "Pillow>=8.0.0",
        "psutil",
        "opencv-python",
        "seaborn",
        "livelossplot",
    ],
)
