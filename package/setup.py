from setuptools import setup

setup(
    name="e3k",  # This is the name of your package
    version="0.1.0",
    author="""
    Kornelia Flizik, Juraj Kret, Max Meiners, Panna Pfandler, Wojciech Stachowiak
    """,
    author_email="your.email@example.com",
    description="""
    Get emotion predictions from audio and video files, train models, and more
    """,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="""
    https://github.com/BredaUniversityADSAI/2023-24d-fai2-adsai-group-nlp3/tree/main
    """,
    # packages=find_packages(where="src"),  # Specify the src directory
    # package_dir={"": "src"},  # Tell setuptools to look in src for packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas >= 2.0",
        "librosa",
        "soundfile",
        "spacy",
        "webrtcvad",
        "whisper",
        "openai-whisper",
        "pydub",
        "tqdm",
        "transformers",
        "tensorflow == 2.10",
        "matplotlib",
        "scikit-learn",
    ],
    package_data={"": ["LICENSE", "README.md"]},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "e3k=e3k.cli:main",  # Command-line entry point
        ],
    },
)
