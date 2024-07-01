from setuptools import setup, find_packages

setup(
    name="SALSA no lr needed",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # List your dependencies here, e.g.,
       # "numpy>=1.18.5",
        "torch>=2.0.0",
    ],
    entry_points={
        'console_scripts': [
            # 'your-command=your_package.module:function',
        ],
    },
    author="Philip Kenneweg",
    author_email="pkenneweg@techfak.uni-bielefeld.de",
    description="A pytorch optimizer that does not need a learning rate",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="hhttps://github.com/TheMody/No-learning-rates-needed-Introducing-SALSA-Stable-Armijo-Line-Search-Adaptation",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
