from setuptools import find_packages, setup

setup(
    name="minine",
    version="0.3",
    author="Marijn De Kerpel",
    author_email="marijn.dekerpel@gmail.com",
    packages=find_packages(),
    install_requires=[],
    package_data={
        "minine": ["minine/modules/config/*", "minine/modules/weights/*"],
    },
)
