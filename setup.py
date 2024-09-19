import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyforestscan",
    version="0.1.3",
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
    python_requires=">=3.8",
    install_requires=[
        'rasterio>=1.3.11',
        'pdal>=3.4.5',
        'geopandas>=1.0.1',
        'pyproj>=3.6.1',
        'shapely>=2.0.6',
        'pandas>=2.2.2',
        'numpy>=2.1.1',
        'matplotlib>=3.9.2',
        'scipy>=1.14.1',
        'mayavi>=4.8.2',
    ],
)
