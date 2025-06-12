from setuptools import setup, find_packages

setup(
    name="napari-sprout",
    version="0.1.0",
    author="SPROUT napari integration",
    description="napari plugin for SPROUT (Semi-automated Parcellation of Region Outputs Using Thresholding and Transformation)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={
        "napari_sprout": ["napari.yaml"],
    },
    python_requires=">=3.8",
    install_requires=[
        "napari>=0.4.16",
        "napari-plugin-engine>=0.2.0",
        "numpy>=1.20.0",
        "scikit-image>=0.18.0",
        "tifffile>=2021.0.0",
        "pandas>=1.3.0",
        "PyYAML>=5.4.0",
        "qtpy",
        "magicgui>=0.3.0",
    ],
    entry_points={
        "napari.manifest": [
            "napari-sprout = napari_sprout:napari.yaml",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Framework :: napari",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
