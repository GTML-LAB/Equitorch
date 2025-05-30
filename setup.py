import setuptools

setuptools.setup(
    name="equitorch",  # Package name set to equitorch
    version="1.0.0",  # Default version
    author="Tong Wang", # Placeholder - Update if needed
    author_email="TongWang_2000@outlook.com", # Placeholder - Update if needed
    description="An efficient modularized package for SO(3)/O(3) equivariant neural networks", # Placeholder description
    long_description="""An efficient modularized package for SO(3)/O(3) equivariant neural networks""", # Placeholder long description
    long_description_content_type="text/markdown",
    url="https://equitorch.readthedocs.io/en/latest/index.html", # Placeholder URL - Update if needed
    packages=setuptools.find_packages(exclude=['test*']), # Automatically find packages in 'equitorch/'
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Placeholder License - Update if needed
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12', # Example Python version requirement
    install_requires=[
        # Add actual dependencies here, for example:
        # 'torch>=2.4.0',
        'triton>=3.2.0',
    ],
)
