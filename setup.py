from setuptools import find_packages, setup
from typing import List
e_dot ='-e .'

def get_file(path:str)-> List[str]:
    r = []
    with open (path) as f:
        r = f.readlines()
        r = [R.replace('\n','') for R in r]
        if e_dot in r:
            r.remove(e_dot)
        return r


setup(name="Wafer Fault Dectection", version='0.1', author="Piyush",
packages=find_packages(), install_requires= get_file('requirements.txt'))