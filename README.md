# pylbsr

My Python toolbox

## Installation

### conda (recommended)

`bedtools`, `zlib`, and `py2bit` are all available prebuilt from conda-forge/bioconda, so
nothing needs to be compiled:

```sh
conda install -c conda-forge -c bioconda bedtools zlib py2bit
pip install .
```

### pip

Works, but some dependencies need to be built from source, which requires a few system
prerequisites first:

- A C/C++ compiler and zlib development headers, for `pybedtools`'s C extension:
  - Debian/Ubuntu: `apt install build-essential zlib1g-dev`
  - Fedora/RHEL: `dnf install gcc-c++ zlib-devel`
  - macOS (Homebrew): `brew install zlib`
- The `bedtools` CLI binary on `PATH` at runtime — this is not a Python package, so `pip`
  cannot install it. Install it via your OS package manager, `conda install bedtools`, or from
  [the bedtools releases page](https://github.com/arq5x/bedtools2/releases).

Once those are in place:

```sh
pip install .
# or directly from GitHub:
pip install git+https://github.com/lambosaur/pylbsr.git
```

### pixi (for contributing to `pylbsr` itself)

A `pixi.toml` is provided as a local dev convenience — it is not required to use the library,
only to work on it:

```sh
pixi install        # dev environment (linting, type-checking, tests) — the default
pixi run test        # run the test suite
pixi install -e bare  # no dev extras, for sanity-checking a plain install
pixi install -e ml    # adds torch (heavy; opt-in only)
```
