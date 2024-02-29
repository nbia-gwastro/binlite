from setuptools import setup, find_packages

MAJOR   = 0
MINOR   = 1
VERSION = f"{MAJOR}.{MINOR}"
# MICRO = 1
# VERSION = f"{MAJOR}.{MINOR}.{MICRO}"

NAME             = "binlite"
AUTHOR           = "Daniel J. D'Orazio"
AUTHOR_EMAIL     = "daniel.dorazio@nbi.ku.dk"
DESCRIPTION      = 'Rapidly generate periodic lightcurves of accreting binaries'
LONG_DESCRIPTION = 'Generate photometric variability templates for equal mass \
                    binaries in co-planar gaseous disks, spanning a continuous \
                    range of orbital eccentricities up to 0.8 for both prograde \
                    and retrograde systems'
# PYTHON_REQUIRES = ">=" + min_version

# =============================================================================
def setup_package():
    metadata = dict(
        name = NAME,
        version = VERSION,
        author  = AUTHOR,
        maintainer = AUTHOR,
        author_email = AUTHOR_EMAIL,
        description  = DESCRIPTION,
        long_description = LONG_DESCRIPTION,
        packages = find_packages(),
        include_package_data = True,
        package_data = {"binlite": ["data/*.dat"]}, # if do this shouldn't need a MANIFEST.in
        license = "MIT",
        platforms = ["Any"],
        # url = GIT_URL,
        # download_url = DOWNLOAD_URL,
        # install_requires = NONE,
        # python_requires = PYTHON_REQUIRES,
    )
    setup(**metadata)

# =============================================================================
if __name__ == "__main__":
    # Running the build is as simple as: 
    # >> python setup.py sdist bdist_wheel
    # This command includes building the required Python extensions
    setup_package()

