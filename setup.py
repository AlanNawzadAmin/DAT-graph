from setuptools import setup, find_packages

setup(
    name='dat_graph',
    description='Code submitted implementing DAT-Graph.',
    packages=find_packages(where='./dat_graph'),
    python_requires='>=3.9,<3.12',
    install_requires=['numpy', 'matplotlib', 'torch', 'sdcd', 'ipykernel']
)