from setuptools import setup, find_packages

setup(
    name='components',
    version='0.0.1',
    packages=["components"],
    install_requires=[
        "numpy",
        "torch",
        "gym==0.14.0",
        "gym-minigrid",
        "pyyaml",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-image",
        "sklearn",
        #"mujoco-py"
        "tensorflow",
        "joblib"
    ]
)
