# -*- coding: utf-8 -*-
"""
"""

from setuptools import setup, find_packages
import runpy
from pathlib import Path

from cslug.building import (build_slugs, bdist_wheel, CSLUG_SUFFIX,
                            copy_requirements)

HERE = Path(__file__).resolve().parent

readme = (HERE / 'README.md').read_text("utf-8")

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
    install_requires=copy_requirements(),
    entry_points={"pyinstaller40": "hook-dirs=motmot:_PyInstaller_hook_dir"},
    extras_require={
        "test": [
            'pytest>=3', 'pytest-order', 'coverage', 'pytest-cov',
            'coverage-conditional-plugin', 'meshzoo'
        ]
    },
    license="MIT license",
    long_description=readme,
    long_description_content_type='text/markdown',
    package_data={
        "motmot": ["*" + CSLUG_SUFFIX, "*.json", "geometry/_unit_vector.pyi"]
    },
    keywords='motmot',
    name='motmot',
    packages=find_packages(include=['motmot', 'motmot.*']),
    url='https://github.com/bwoodsend/motmot',
    version=runpy.run_path(HERE / "motmot/_version.py")["__version__"],
    zip_safe=False,
    cmdclass={
        "build": build_slugs("motmot._slug:slug"),
        "bdist_wheel": bdist_wheel,
    },
)
