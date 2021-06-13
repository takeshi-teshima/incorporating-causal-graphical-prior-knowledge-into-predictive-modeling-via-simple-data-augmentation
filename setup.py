from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name="causal-data-augmentation",
    version="1.0.0",
    description='Implementation of Causal Data Augmentation.',
    long_description=readme,
    author='',
    author_email='',
    url='',
    license=license,
    packages=find_packages(exclude=('docs', 'experiments')),
)
