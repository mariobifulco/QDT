import sys

from .campaign import main

if __name__ == "__main__":
    main(["collect", *sys.argv[1:]])
