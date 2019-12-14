from setuptools import setup, find_packages

setup(
    name='novelty_guided_package',
    version='0.0.1',
    packages=["novelty_guided_package"],
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
        #"mujoco-py",
        "tensorflow",
        "joblib",
        "tqdm"
    ]
)
