# setup.py
from setuptools import setup, find_packages

setup(
    name="bb_meta_imitation_learning",
    version="0.1.0",
    description="BC & meta-imitation learning for maze tasks",
    packages=find_packages(include=[
        "bb_meta_imitation_learning",
        "bb_meta_imitation_learning.*",
    ]),
    install_requires=[
        "torch",
        "numpy",
        "gymnasium",
        "pygame",
        "numba",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "bc-pretrainer=bb_meta_imitation_learning.bc_pre_trainer.main:main",
        ],
    },
    python_requires=">=3.7",
)
