from setuptools import setup

setup(
    name='nlpapi',
    version='1.0',
    long_description=__doc__,
    zip_safe=False,
    install_requires=[
        'flask',
        'nltk',
        'numpy',
        'spacy',
        'beautifulsoup4',
        'stanfordcorenlp',
        'requests'
    ]
)