from setuptools import setup, find_packages

setup(
    name="ktamp_learning",
    version="0.1.0",
    description="Learning-based approach for knowledge transfer in robotic task and motion planning",
    author="",
    author_email="",
    packages=find_packages(include=["ktamp_learning", "ktamp_learning.*", "src", "src.*"]),
    install_requires=[
        "torch",
        "torchvision",
        "pytorch-lightning",
        "hydra-core",
        "omegaconf",
        "numpy",
        "opencv-python",
        "scipy",
        "matplotlib",
        "tqdm",
    ],
    python_requires=">=3.8",
)
