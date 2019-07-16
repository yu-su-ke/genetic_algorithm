# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='GeneticAlgorithm',
    version='0.0.1',
    description='',
    long_description=readme,
    author='mokky',
    author_email='',
    install_requires=['numpy', 'matplotlib', 'pandas', 'scipy', 'pymongo'],
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    test_suite='tests'
)