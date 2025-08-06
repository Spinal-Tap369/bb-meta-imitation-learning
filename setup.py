from setuptools import setup, find_packages

setup(
    name="bb_meta_imitation_learning",
    version="0.1.0",
    description="Meta-imitation learning framework with BC pre-training, meta-training, and DAgger",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.7.0",
        "numpy>=2.2.5",
        "gymnasium>=1.1.1",
        "pygame>=2.6.1",
        "numba>=0.61.2",
        "tqdm>=4.67.1",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "bc-pretrainer=bb_meta_imitation_learning.bc_pre_trainer.main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
