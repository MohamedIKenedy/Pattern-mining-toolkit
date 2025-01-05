from setuptools import setup, find_packages

setup(
    name='pattern-mining-toolkit',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A toolkit for various pattern mining algorithms.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/pattern-mining-toolkit',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'networkx'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)