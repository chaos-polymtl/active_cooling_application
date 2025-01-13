'''
Copyright 2024-2025, the Active Cooling Experimental Application Authors

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

from setuptools import setup, find_packages

# Function to read the requirements.txt file and return a list of dependencies
def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename, 'r') as file:
        return [line.strip() for line in file if line.strip() and not line.startswith('#')]

setup(
    name="active_cooling_experimental",
    version="0.1.0",
    description="Project destinated to store code and scheme for experimental setup of the active cooling project.",
    author="Bruno Blais",
    author_email="bruno.blais@polymtl.ca",
    url="https://github.com/chaos-polymtl/active_cooling_experimental",
    packages=find_packages(include=["*","source", "source.*"]),
    include_package_data=True,
    package_data={
        'source': ['style.qss', 'nrc.png'],         # Add style.qss here
    },
    install_requires=parse_requirements('requirements.txt'),  # Load dependencies from requirements.txt
    entry_points={
        'console_scripts': [
            'active-cooling-experiment=source.main:Application.run',  # Entry point for command line execution
            'active-cooling-experiment-test=source.main:Application.run_test'
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.8'
)
