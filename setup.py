from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cardquant",
    version="0.1.2",
    author="Evan Semet",
    author_email="evancsemet@gmail.com",
    description="A library for mock quantitative trading exercises.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/evansemet/cardquant",
    project_urls={
        "Bug Tracker": "https://github.com/evansemet/cardquant/issues",
        "Documentation": "https://github.com/evansemet/cardquant#readme",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pandas",
        "typing_extensions; python_version < '3.9'",
    ],
    include_package_data=True,
    keywords="quantitative trading, options pricing, card games, finance, simulation",
    license="MIT",
)
