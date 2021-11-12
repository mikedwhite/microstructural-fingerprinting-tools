[![Documentation Status](https://readthedocs.org/projects/microstructural-fingerprinting-tools/badge/?version=latest)](https://microstructural-fingerprinting-tools.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


# Microstructural Fingerprinting Tools
Python package for constructing compressed representations of microstructural image data.


## Examples

Examples provided operate on a 3-class subset of the Carnegie Mellon University ultrahigh carbon steel (CMU-UHCS)
dataset, which can be downloaded [here](https://materialsdata.nist.gov/handle/11256/940).

Two main approaches for feature extraction are considered. Namely, visual bag of words (VBOW) and convolutional neural
networks (CNN). There are two options for CNN fingerprints (flattening or max-pooling of final convolution layer into
single vector). `cnn_fingerprints.py` provides an example with either AlexNet or VGG for feature extraction and
flattening or max-pooling for fingerprint construction.

VBOW operates on base features. These can either come from keypoint-based methods, such as the scale-invariant feature
transform (SIFT), or from CNN features output from the final convolution layer. `vbow_fingerprints.py` provides an
example with either SIFT, SURF or CNN base features. Number of clusters and order of fingerprints can also be specified.


## Documentation

Full documentation can be found on Read the Docs [here](https://microstructural-fingerprinting-tools.readthedocs.io/en/latest/).


## Acknowledgements

This project was developed at the University of Manchester with funding from the
Engineering and Physical Sciences Research Council (EPSRC) grants EP/S022635/1 and EP/N510129/1 and Science Foundation
Ireland (SFI) grant 18/EPSRC-CDT/3584.


## License (Modified BSD)

Copyright (c) 2021, University of Manchester.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
