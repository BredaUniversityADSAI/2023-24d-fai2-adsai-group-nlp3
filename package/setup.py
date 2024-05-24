from setuptools import setup, find_packages

setup(
    name="package_name",  # This is the name of your package
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Get emotion predictions from audio and video files, train models, and more",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/BredaUniversityADSAI/2023-24d-fai2-adsai-group-nlp3/tree/main",
    packages=find_packages(where='src'),  # Specify the src directory
    package_dir={'': 'src'},  # Tell setuptools to look in src for packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        "numpy",
        "pandas >= 2.0"
    ],
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.rst"],
        "my_package": ["data/*.dat"],  # Example of including additional data files in the package
    },
    entry_points={
        'console_scripts': [
            'package_name=package_name.cli:main',  # Command-line entry point
        ],
    },
)