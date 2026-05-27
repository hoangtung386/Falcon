from pathlib import Path

from setuptools import find_packages, setup

FILE = Path(__file__).resolve()
PARENT = FILE.parent
README = (PARENT / "README.md").read_text(encoding="utf-8")
REQUIREMENTS = [
    line.strip() for line in (PARENT / "requirements.txt").read_text().strip().splitlines()
    if line.strip() and not line.startswith("#")
]


exec((PARENT / "falcon" / "version.py").read_text())
setup(
    name="falcon-age-gender",
    version=__version__,  # version of pypi package # noqa: F821
    python_requires=">=3.8",
    description="Falcon - Multi-input Transformer for Age and Gender Estimation",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/anomalyco/Falcon",
    project_urls={"Datasets": "https://wildchlamydia.github.io/lagenda/"},
    author="Falcon Contributors",
    author_email="tung@anomaly.co",
    packages=find_packages(include=["falcon", "falcon.*", "tools", "tools.*"]),
    include_package_data=True,
    install_requires=REQUIREMENTS,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: Attribution-ShareAlike 4.0",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
    ],
    keywords="machine-learning, deep-learning, vision, ML, DL, AI, transformer, falcon, age-estimation, gender-recognition",
)
