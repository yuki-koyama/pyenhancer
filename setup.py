import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyenhancer",
    version="1.0dev1",
    author="Yuki Koyama",
    author_email="yuki@koyama.xyz",
    description="A tiny NumPy-based library for image color enhancement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yuki-koyama/pyenhancer",
    packages=["pyenhancer"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "scikit-image",
    ],
)
