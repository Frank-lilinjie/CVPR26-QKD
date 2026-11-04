"""Allow ``python -m qkd_mindspore`` to launch the MindSpore trainer."""

from .main import main


if __name__ == "__main__":
    raise SystemExit(main())
