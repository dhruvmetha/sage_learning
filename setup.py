from setuptools import setup, find_packages

setup(
    name="ktamp-learning",
    version="0.1.0",
    description="Learning-based approach for knowledge transfer in robotic task and motion planning",
    author="",
    author_email="",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pytorch-lightning",
        "hydra-core",
        "numpy",
        "tqdm",
    ],
    python_requires=">=3.6",
)
