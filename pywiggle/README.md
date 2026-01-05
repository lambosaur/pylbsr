# README - Onboarding for PARNET projects

## Set-up instructions - General

Here are some set-up instructions to get you started with any of the repositories, be it this one (read-only)
or other repositories that you will create for your own projects.

### SSH keys

```bash
ssh-keygen -t ed25519 -C "your_email@example.com" (on cluster)
cat ~/.ssh/id_ed25519.pub
# copy the public key into github ssh settings
```

Use-cases:

- Pushing / pulling changes from private repositories
- Installing private packages from GitHub (e.g. group-specific private repositories for PARNET)

## Set-up instructions - parnet_demo

```bash
git clone https://github.com/ml4rg25-parnet/parnet_demo.git
cd parnet_demo
```

### Virtual environment

#### Installation using pixi

```bash
pixi install -e "parnet-demo"
pixi install -e "parnet-gpu-demo"
```

Then "activate" the environment: `pixi shell -e "parnet-demo"` or `pixi shell -e "parnet-gpu-demo"`.

#### Installation using Conda

> [!NOTE]
> conda YAML files are generated using `pixi-pack` ; see the [Updating the environment](#updating-the-environment) section below.

#### Updating the environment

> [!WARNING]
> We stronly advise using the `pixi` package manager to manage the environment.
> If you are using `conda` to create the environment and update it, please consider propagating the changes to the
> `pixi.toml` and `pyproject.toml`.


##### Add a package

Updates to the `pixi.toml` can either be done manually or using the `pixi` commands (e.g. `pixi add <package>`).

If the `pixi add <package>` command fails owing to dependencies resolution issues, one alternative is to manually
update the `pixi.toml` with the following steps:

1. add the package in the relevant section (conda or pypi) with free version constraints (e.g. `jax = "*"`).
2. run `pixi update` to have the modification applied.
3. get the installed version of the package with `pixi run conda list <package>` or `pixi run pip show <package>`.
4. update the `pixi.toml` with the resolved version (e.g. `jax = "==0.3.2"`).

##### Propagate changes

After any modification, proceed through the following steps (hereafter with the example of the `parnet-demo` environment):

1. manually update the `pyproject.toml` file with the newly added, version-resolved package.

> [!IMPORTANT] Keeping the `pyproject.toml` part is relevant for repositories that you plan to install as dependencies in other projects.
> e.g. with the `parnet` repository, which is a dependency of the `parnet_demo` repository.
> You can ignore this step if you are not planning to use the repository as an installable package.

2. run the `pixi-pack` command to produce a tarball of the environment.

  ```bash
  # Here with the example of the CPU-only environment for linux-64 ; you can generate multiple tarballs for different platforms.
  pixi-pack pack --environment parnet-demo --ignore-pypi-errors --platform linux-64 --output-file envs/pixi-pack.parnet-demo.linux-64.tar.gz ./pixi.toml
  ```

> [!WARNING]
> Unfortunately, some of the packages in the `pixi.toml` are not available on PyPI.
> You can bypass these by using the `--ignore-pypi-errors`.
> so you will need to manually install them in the conda environment, using `pip install git+https://github.com/<organization>/<repo>.git@<rev>`
> All the packages to install would be under the `[*.pypi-dependencies]` sections of the `pixi.toml` file.

3. Untar the archive:

  ```bash
  cd envs/ && mkdir -p pixi-pack.parnet-demo.linux-64/ && tar -xf pixi-pack.parnet-demo.linux-64.tar.gz -C pixi-pack.parnet-demo.linux-64/
  ```

4. Copy the `environment.yml` file from the unpacked directory into the `envs/` directory: `cp envs/pixi-pack.parnet-demo.linux-64/environment.yml envs/pixi-pack.parnet-demo.linux-64.conda.yml`

5. Edit the copy `envs/pixi-pack.parnet-demo.linux-64.conda.yml` to apply the following changes:
    - replace the channels section with the channels list (available with `pixi config list`)
    - under the `- pip` section, remove the two lines `- --no-index` and `- --find-links ./pypi`

This resulting yaml file can then be used to create a new conda environment with:

```bash
conda env create -f envs/pixi-pack.parnet-demo.linux-64.conda.yml -n parnet-demo
```

> [!WARNING]
> Again: please consider using `pixi` to manage the environment.


If updating your environment through `conda`, you can export the currently activated environment to a lock file with:

```bash
conda env export --from-history --no-builds -n $(ENV_NAME) -f environment.lock.yml
```


### Datasets

From a terminal, download the datasets using the `gdown` python library (NOTE: the datasets Google Drive
folder was set to be publicly accessible, so you should not need to authenticate).

Store (or symlink) the provided datasets into the `resources/` directory. The expected structure is as follows:

```bash
./resources/
├── general/
├── parnet_encore_eclip/
├── cell_line_rnaseq/
├── parnet_models/
└── sun_et_al_2021_icshape_data/
```
