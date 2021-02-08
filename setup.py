# -*- coding: utf-8 -*-
"""
"""

from setuptools import setup, find_packages
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent

readme = (HERE / 'README.rst').read_text("utf-8")

setup(
    author="Brénainn Woodsend",
    author_email='bwoodsend@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="A sophisticated mesh class for analysing 3D surfaces.",
    install_requires=[],
    extras_require={
        "test": [
            'pytest>=3', 'pytest-order', 'coverage', 'pytest-cov',
            'coverage-conditional-plugin'
        ]
    },
    license="MIT license",
    long_description=readme,
    package_data={"motmot": []},
    keywords='motmot',
    name='motmot',
    packages=find_packages(include=['motmot', 'motmot.*']),
    url='https://github.com/bwoodsend/motmot',
    version=runpy.run_path(HERE / "motmot/_version.py")["__version__"],
    zip_safe=False,
)
