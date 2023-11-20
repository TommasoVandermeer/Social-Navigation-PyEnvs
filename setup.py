from setuptools import setup


setup(
    name='social_nav',
    version='0.0.1',
    packages=[
        'crowd_nav',
        'crowd_nav.configs',
        'crowd_nav.policy',
        'crowd_nav.utils',
        'social_gym',
        'social_gym.src',
        'social_gym.src.policy',
    ]
)