import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

        setuptools.setup(
                name="xla_add",
                version="0.0.1",
                author="Vaibhav Singh",
                author_email="sivaibhav@google.com",
                description="Experimental snippets for MP Training",
                long_description=long_description,
                long_description_content_type="text/markdown",
                url="https://github.com/xx/yy",
            classifiers=[
                        "Programming Language :: Python :: 3",
                        "License :: OSI Approved :: MIT License",
                        "Operating System :: OS Independent",

            ],
                package_dir={"": "."},
                packages=setuptools.find_packages(exclude=('src')),
                python_requires=">=3.6",

        )
