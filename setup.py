import setuptools

with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

setuptools.setup(
    name='microstructural-fingerprinting-tools',
    version='0.0.2',
    author='Mike White',
    author_email='michael.white-3@postgrad.manchester.ac.uk',
    description='Tools for constructing fingerprint vectors from microstructural image data',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/mikedwhite/microstructural-fingerprinting-tools',
    license='Modified BSD',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
