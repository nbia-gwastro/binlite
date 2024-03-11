from setuptools import setup, find_packages

MAJOR = 1
MINOR = 0
MICRO = 0
VERSION = f"{MAJOR}.{MINOR}.{MICRO}"

NAME             = "binlite"
AUTHOR           = "Christopher Tiede & Daniel D'Orazio"
AUTHRO_EMAIL     = "22056557+ctiede@users.noreply.github.com"
DESCRIPTION      = "Rapidly generate periodic lightcurves of accreting binaries"
LONG_DESCRIPTION = "Generate photometric variability templates for equal mass \
                    binaries in co-planar gaseous disks, spanning a continuous \
                    range of orbital eccentricities up to 0.8 for both prograde \
                    and retrograde systems"
GIT_URL = "https://github.com/nbia-gwastro/binlite"

# =============================================================================
def setup_package():
    metadata = dict(
        name = NAME,
        version = VERSION,
        author  = AUTHOR,
        maintainer = AUTHOR,
        author_email = AUTHRO_EMAIL,
        description  = DESCRIPTION,
        long_description = LONG_DESCRIPTION,
        include_package_data = True,
        packages = find_packages(),
        package_data = {"binlite": ["data/*.dat"]},
        license = "MIT",
        platforms = ['Any'],
        url = GIT_URL,
        install_requires = ['setuptools', 'numpy', 'scipy'],
    )
    setup(**metadata)

# =============================================================================
if __name__ == "__main__":
    # Running the build is as simple as: 
    # >> python setup.py sdist bdist_wheel
    # This command includes building the required Python extensions
    setup_package()

