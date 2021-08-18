from setuptools import setup

setup(
    name='rbm_adapter',
    version='0.0.1',
    description='Package for interfacing with the C RBM program',
    license='GNU',
    packages=['rbm'],
    author='David Bucher',
    author_email='David.Bucher@physik.lmu.de',
    keywords=['spin liquids', 'tksvm', 'mlgen'],
    url='https://gitlab.physik.uni-muenchen.de/David.Bucher/rbm-nqs',
    scripts=['scripts/ked']
)
