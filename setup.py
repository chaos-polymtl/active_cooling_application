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
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.8'
)
