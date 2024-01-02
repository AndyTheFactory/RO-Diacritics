from setuptools import setup, find_packages

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setup(
    # $ pip install sampleproject
    name='ro-diacritics',  # Required

    version='0.9.4',  # Required

    description='Python API for Romanian diacritics restoration',  # Required

    long_description=long_description,
    long_description_content_type="text/markdown",
    maintainer="Andrei Paraschiv",
    maintainer_email="andrei@thephpfactory.com",
    author="Andrei Paraschiv",
    author_email="andrei@thephpfactory.com",

    url='https://github.com/AndyTheFactory/RO-Diacritics',  # Optional

    classifiers=[  # Optional
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Human Machine Interfaces',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Filters',
        'Topic :: Text Processing :: General',
        'Topic :: Text Processing :: Indexing',
        'Topic :: Text Processing :: Linguistic',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3'
    ],

    keywords='romanian diacritcs language restoration diacritice python',

    # package_dir={"": "."},
    packages=find_packages(),

    install_requires=['torch', 'torchtext', 'numpy', 'tqdm', 'nltk', 'scikit-learn',],

    zip_safe=False,
)
