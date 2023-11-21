from setuptools import setup, find_packages

setup(
    name='social_nav',
    version='0.0.1',
    packages= find_packages(),
    package_data={'social_gym': ['fonts/Roboto-Black.ttf'],
                  'crowd_nav': ['configs/']},
    include_package_data=True,
)