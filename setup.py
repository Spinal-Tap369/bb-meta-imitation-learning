# setup.py

from setuptools import setup, find_packages

setup(
    name="bb_meta_imitation_learning",
    version="0.1.0",
    description="BC & meta-imitation learning for maze tasks",
    packages=find_packages(include=["bb_meta_imitation_learning", "bb_meta_imitation_learning.*"]),
    include_package_data=True,
    package_data={
        "bb_meta_imitation_learning.env": ["img/**/*", "img/*"],
    },
    install_requires=[
        "numpy", "gymnasium", "pygame", "numba", "tqdm", "torch"
    ],
    entry_points={
        "console_scripts": [
            "bc-pretrainer=bb_meta_imitation_learning.bc_pre_trainer.main:main",
            "bc-meta-train=bb_meta_imitation_learning.bc_meta_train.train:run_training",
        ],
    },
    python_requires=">=3.7",
)
