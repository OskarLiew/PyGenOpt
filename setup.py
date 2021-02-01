import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setuptools.setup(
    name="genopt",
    version="0.1.0",
    author="Oskar Liew",
    author_email="oskar@liew.se",
    description="Simple genetic optimization package for python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OskarLiew/PyGenOpt",
    packages=setuptools.find_packages(),
    license="LICENSE",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=required,
)
