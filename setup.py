from setuptools import find_packages, setup

setup(
    name='jagpascoe_ML_toolkit',
    #packages=find_packages(),
    packages=find_packages(include=['jagpascoe_ML_toolkit'],exclude=["tests", "tests.*"]),    
    version='0.1.0',
    description='My first Python library',
    author='Me',
    license='AGP',
)