# Deprecated: disease-specific demo runner removed.
# This file remains only as a stub to avoid import errors in older scripts.
# Please use the generic CLI under `tools/cli/cli.py`.


def main():
    raise RuntimeError(
        "The disease-specific demo has been removed. Use `python -m tools.cli.cli --help`. "
    )


if __name__ == "__main__":
    main()
