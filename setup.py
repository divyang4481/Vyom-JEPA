from setuptools import setup, find_packages

setup(
    name="vyom_jepa",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0",
        "numpy",
        "pyyaml",
        "timm",
        "transformers",
        "sentence-transformers",
        "bitsandbytes",
    ],
)
