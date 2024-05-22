import setuptools

if __name__=="__main__":

    with open("README.md", "r") as fh:
        long_description = fh.read()

    setuptools.setup(
        name='clip-mmd',
        version="0.0.1",
        author="arr2",
        author_email="whnahengyuan@gmail.com",
        description="",
        long_description=long_description,
        long_description_content_type="text/markdown",
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
        url="https://github.com/GaParmar/clean-fid",
        packages=['clip-mmd'],
        include_package_data=True,
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: BSD License",
            "Operating System :: OS Independent",
        ],
    )