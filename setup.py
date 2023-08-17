from setuptools import setup, find_packages

setup(
    name="maddpkg",
    version="0.3.0",
    description="A Python package to compute the MADD metric",
    url="https://github.com/melinaverger/MADDpkg",
    author="Mélina Verger",
    author_email="melina.verger@lip6.fr",
    package_dir={"":"src"},
    packages=find_packages(where="src"),
    license="CC BY-NC 4.0",
    classifiers=[
        "Programming Language :: Python :: 3.10.4",
        "Operating System :: OS Independent"
        ],
    install_requires=["scikit-learn", "pandas", "numpy"],
    python_requires=">=3.10.4"
)