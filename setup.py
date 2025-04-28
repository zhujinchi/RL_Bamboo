from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rl_bamboo_rejoin",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Game-Theoretic Reinforcement Learning Framework for Ancient Bamboo Slip Fragment Matching",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rl_bamboo_rejoin",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/rl_bamboo_rejoin/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "scipy>=1.5.0",
        "tqdm>=4.50.0",
    ],
    keywords="reinforcement learning, game theory, archaeology, bamboo slips",
)