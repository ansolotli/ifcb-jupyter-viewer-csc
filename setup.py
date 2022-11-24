from setuptools import setup, find_packages

with open("requirements.txt") as fh:
    requirements = [line.strip() for line in fh if not line.startswith("#")]

setup(
    name="ifcb-jupyter-viewer",
    version="0.0.1",
    packages=find_packages(),
    description="Graphical tool for viewing automatically classified IFCB images",
    author="Otso Velhonoja",
    url="https://github.com/veot/ifcb-jupyter-viewer",
    install_requires=requirements,
)
