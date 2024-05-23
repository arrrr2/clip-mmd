import setuptools
from setuptools import find_packages

if __name__=="__main__":

    with open("README.md", "r") as fh:
        long_description = fh.read()

    setuptools.setup(
        name='clip-mmd',
        version="0.0.2",
        author="arr2",
        author_email="whnahengyuan@gmail.com",
        packages=find_packages(),
        description="CLIP Maximum Mean Discrepancy (CMMD) for evaluating generative models",
        license='Apache License 2.0',
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/arrrr2/clip-mmd",
        install_requires=[
            "torch",
            "torchvision",
            "numpy",
            "scipy",
            "scikit-image",
            "tqdm",
            "pillow",
            "requests",

        ],
        include_package_data=True,
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
        ],
        entry_points={
            'console_scripts': [
                'clip-mmd=clip_mmd.cli:main', 
            ],
        },
    )