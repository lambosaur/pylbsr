import os
from pathlib import Path


def find_project_root_from_notebook_path(expected_notebook_rel: os.PathLike) -> Path:
    """Determine the project root based on the expected path of a notebook relative to that root.

    Args:
        expected_notebook_rel (os.PathLike): Notebook path relative to the project root,
            e.g., "notebooks/subdir/my_notebook.ipynb".

    Returns:
        Path: The detected project root.

    Raises:
        FileNotFoundError: If the notebook cannot be found from CWD or VSCode variable.
    """
    expected_notebook_rel = Path(expected_notebook_rel)
    cwd = Path.cwd().resolve()

    # First try: is the notebook reachable from current cwd?
    candidate_notebook = cwd / expected_notebook_rel
    if candidate_notebook.exists():
        return cwd

    # Second try: use VSCode notebook variable if available
    if "__vsc_ipynb_file__" in globals():
        vscode_path = Path(globals()["__vsc_ipynb_file__"]).resolve()
        candidate_root = vscode_path.parent
        # Compute candidate root such that candidate_root / expected_notebook_rel exists
        # Walk up folders to match relative path length
        parts_to_strip = len(expected_notebook_rel.parts) - 1
        for _ in range(parts_to_strip):
            candidate_root = candidate_root.parent

        candidate_notebook = candidate_root / expected_notebook_rel
        if candidate_notebook.exists():
            return candidate_root

    # Last try: naive search from CWD
    for parent in [cwd, *list(cwd.parents)]:
        candidate_notebook = parent / expected_notebook_rel
        if candidate_notebook.exists():
            return parent

    raise FileNotFoundError(
        f"Cannot locate notebook '{expected_notebook_rel}' from CWD ({cwd}) or VSCode variables."
    )


def assert_notebook_working_dir(expected_local_file: os.PathLike) -> Path:
    """Assert/update the working directory to the notebook's parent, for relative paths references.

    This function is used in a set-up where notebooks are contained within a project
    directory structure in which we want to reference filepaths relative to the notebook.
    e.g. "../src" or "../resources" should be accessible if the notebook is in
    "../notebooks/<notebook_name>.ipynb".

    The function first check the filepath of the expected local file relative
    to the current working directory.

    If not found, the function will try to use the VSCode Jupyter variable `__vsc_ipynb_file__`
    which should report the path of the notebook file being executed.

    It then checks if the expected local file exists, relative to the new working directory.


    Args:
        expected_local_file (os.PathLike): The expected local file to check for in the current working directory.
            This can be the name of the notebook file.

    Raises:
        KeyError: if the `__vsc_ipynb_file__` variable is not found in the global scope, while the first CWD check failed.
        FileNotFoundError: if the expected local file is not found in the current working directory after attempting to change it.
    """
    import os
    from pathlib import Path

    # Find file in current and subdirectories. Stop at first find
    def find_file(filename: str, search_path: str) -> str | None:
        for root, dirs, files in os.walk(search_path):
            if filename in files:
                return os.path.join(root, filename)
        return None

    cwd = Path(os.getcwd())

    expected_local_filepath = cwd / expected_local_file

    if expected_local_filepath.exists():
        # If the expected file exists but is not at the root of the CWD,
        # change the CWD to the directory containing the expected file.
        if expected_local_filepath.resolve().is_relative_to(cwd):
            os.chdir(expected_local_filepath.parent)
            print(f"Changed CWD to {expected_local_filepath.parent}")

            return expected_local_filepath.parent

    else:
        if "__vsc_ipynb_file__" in globals():
            os.chdir(Path(globals()["__vsc_ipynb_file__"]).parent)
            cwd = Path(os.getcwd())
            print(f"Changed CWD to {cwd}")

            # Verify that __vsc_ipynb_file__ folder contains expected file.
            expected_local_filepath = cwd / expected_local_file
            if not expected_local_filepath.exists():
                raise FileNotFoundError(
                    f"Updated (using __vsc_ipynb_file__) CWD: {cwd} ; CWD does not contain expected file."
                )

            return cwd

        else:
            # Search from current directory
            found_file = find_file(expected_local_file, cwd)

            if found_file is None:
                raise ValueError(
                    f"Detected CWD: {cwd} ; CWD does not contain expected file. "
                    "Cannot use __vsc_ipynb_file__ to recover."
                    "Tried locating the file in subdirectories but not found."
                )

            found_file = Path(found_file)
            if found_file.resolve().is_relative_to(cwd):
                os.chdir(found_file.parent)
                print(f"Changed CWD to {found_file.parent}")
                return found_file.parent

            else:
                raise ValueError("Unexpected error: found_file is not relative to cwd")


