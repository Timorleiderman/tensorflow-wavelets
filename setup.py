import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tensorflow-wavelets",
    version="1.1.2",
    author="Timor Leiderman",
    author_email="Timorleiderman@gmail.com",
    description="Tensorflow wavelet Layers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Timorleiderman/tensorflow-wavelets",
    project_urls={
        "Bug Tracker": "https://github.com/Timorleiderman/tensorflow-wavelets/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords = ['Wavelets', 'Tensorflow'],
    install_requires=[            
          'tensorflow',
          'tensorflow-probability',
          'PyWavelets',
      ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
