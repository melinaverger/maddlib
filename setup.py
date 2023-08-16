from setuptools import setup

setup(
    name="maddpkg",
    version="0.0.1",
    description="A Python package to compute the MADD metric",
    url="https://github.com/melinaverger/MADDpkg",
    author="Mélina Verger",
    license="CC BY-NC 4.0",
    classifiers=[
        "Programming Language :: Python :: 3.10.4",
        "Operating System :: OS Independent"
        ],
    install_requires=["scikit-learn", "pandas"],
    python_requires=">=3.10.4"
)