from setuptools import setup, find_packages

setup(
    name='components',
    version='0.0.1',
    packages=["components"],
    install_requires=[
        "numpy",
        "torch",
        "gym",
        "gym-minigrid",
        "pyyaml",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-image",
        "sklearn",
        "mujoco-py"
    ]
)
