import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyforestscan",
    version="0.3.5",
    author="Joseph Emile Honour Percival",
    author_email="ipercival@gmail.com",
    description="Analyzing forest structure using aerial LiDAR data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iosefa/PyForestScan",
    project_urls={
        "Bug Tracker": "https://github.com/iosefa/PyForestScan/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.11",
    install_requires=[
        'requests>=2.32.4',
        'rasterio>=1.3.11',
        'pdal>=3.4.5',
        'geopandas>=1.1.1',
        'pyproj>=3.7.1',
        'shapely>=2.1.1',
        'pandas>=2.3.0',
        'numpy>=2.3.1',
        'matplotlib>=3.10.3',
        'scipy>=1.16.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'flake8>=6.0.0',
            'black>=23.0.0',
        ],
    },
)
