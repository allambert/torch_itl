import os
from setuptools import setup, find_packages

version = None
with open(os.path.join('torch_itl', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not
            line.startswith("#")]


install_reqs = parse_requirements('./requirements.txt')
reqs = [str(ir) for ir in install_reqs]

setup(name='torch_itl',
      version=version,
      description='pytorch compatible integral loss minimization',
      author='Alex Lambert',
      author_email='alex.lambert@protonmail.com',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      install_requires=reqs,
      zip_safe=False)
