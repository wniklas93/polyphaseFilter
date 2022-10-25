from setuptools import setup, find_packages

setup(
   name='polyphaseFilter',
   version='0.1.0',
   author='Niklas Winter',
   author_email='niklas.winter93@googlemail.com',
   packages=find_packages(include=['polyphaseFilter', 'polyphaseFilter.*']),
   description='Module that can be used to generate polyphase filters (FIR/IIR)',
   install_requires=[
       "numpy",
       "pytest",
       "matplotlib",
       "scipy"
   ],
)
