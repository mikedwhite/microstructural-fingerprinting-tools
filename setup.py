import setuptools

with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

with open('requirements.txt', encoding='utf-8') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

setuptools.setup(
    name='microstructural-fingerprinting-tools',
    version='0.0.1',
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
    install_requires=INSTALL_REQUIRES,
    python_requires='>=3.6',
)
