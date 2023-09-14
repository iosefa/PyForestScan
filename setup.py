import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyforestscan",
    version="0.0.1",
    author="Iosefa Percival",
    author_email="ipercival@gmail.com",
    description="UAV image analysis for forest ecology",
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
    install_requires=[
        'setuptools~=65.5.1',
        'rasterio~=1.3.8',
        'pdal~=3.2.3',
        'geopandas~=0.13.2',
        'pyproj~=3.6.0',
        'shapely~=2.0.1',
        'pandas~=2.0.3',
        'numpy~=1.25.1',
        'matplotlib~=3.7.2',
        'mayavi~=4.8.1'
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8"
)
