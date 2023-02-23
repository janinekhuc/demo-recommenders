from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = [line.rstrip() for line in f]

setup(
    name='recommenders',
    version=0.1,
    install_requires=requirements,
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires=">=3.9"
)
