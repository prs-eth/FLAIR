from setuptools import setup, find_packages

setup(
    name="flair",
    version="0.1.0",
    author="Julius Erbach",
    author_email="julius.erbach@gmail.com",
    description="Solving Inverse Problems with FLAIR",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    python_requires=">=3.7",
    keywords="pytorch variational posterior sampling deep learning",
)
