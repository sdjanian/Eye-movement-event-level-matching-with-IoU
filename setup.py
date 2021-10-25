from setuptools import setup, find_packages

print(find_packages())
setup(
    author="Shagen Djanian",
    author_email="s.d@hotmail.dk",
    name='eventMatchingIoU',
    license="MIT",
    description='eventMatchingIoU is a package to measure intersection over unionen for eye movements',
    version='v1',
    url='https://github.com/Kongskrald/Eye-movement-event-level-matching-with-IoU',
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=['numpy>=1.14.5','pandas>=0.23.4'],
    include_package_data=True,
)