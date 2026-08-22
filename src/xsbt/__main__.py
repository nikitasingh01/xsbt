"""Allow ``python -m xsbt`` as well as the installed ``xsbt`` console script."""

from xsbt.cli import main

if __name__ == "__main__":
    main()
