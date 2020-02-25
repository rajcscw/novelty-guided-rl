from setuptools import setup, find_packages

setup(
    name='novelty_guided_package',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        "opencv-python",
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
        #"mujoco-py<2.1,>=2.0",
        "tensorflow",
        "joblib",
        "tqdm"
    ]
)
