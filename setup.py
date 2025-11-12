import setuptools 

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ClinicalTranformer",
    version="0.0.1",
    author="Gustavo Arango",
    author_email="",
    description="A transformer classifier for survival analysis on clinical data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(include=["xai*", "ctv2*"]),
    include_package_data=True,
    install_requires = [],
    extras_require={
        "ctv2": [
            "torch>=2.2.0",
            "accelerate>=0.28.0",
            "anndata>=0.10.0",
            "transformers>=4.40.0",
            "tokenizers>=0.15.2",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Restricted",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points=''' '''
)
