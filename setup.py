from setuptools import setup, find_packages

setup(
    name="ktamp_learning",
    version="0.1.0",
    description="Learning-based approach for knowledge transfer in robotic task and motion planning",
    author="",
    author_email="",
    packages=find_packages(include=["ktamp_learning", "ktamp_learning.*", "src", "src.*"]),
    install_requires=[
        "torch>=2.1",
        "torchvision",
        "lightning>=2.0",
        "hydra-core>=1.3",
        "omegaconf",
        "numpy",
        "opencv-python",
        "scipy",
        "matplotlib",
        "tqdm",
        "tensorboard",
        "torchmetrics",
    ],
    extras_require={
        "fb": [
            "flow-matching",  # Facebook's flow matching library
            "torchdyn",       # For adaptive ODE solvers (dopri5, tsit5)
        ],
    },
    python_requires=">=3.9",
)
