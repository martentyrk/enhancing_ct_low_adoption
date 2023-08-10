from setuptools import setup, find_packages

setup(name='dpfn',
      version='0.1',
      description='',
      url='',
      author='anonymous',
      author_email='anonymous@gmail.com',
      license='LICENSE.txt',
      install_requires=[
          'numpy',
          'scipy',
          'sklearn',
          'matplotlib',
      ],
      packages=find_packages(exclude=('tests')),
      zip_safe=False)
