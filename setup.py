import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lstm-variants",
    version="0.1.2",
    author="Urchade Zaratiana",
    author_email="urchade.zaratiana@gmail.com",
    description="Long-Short Term Memory variants",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/urchade/lstm-variants.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)