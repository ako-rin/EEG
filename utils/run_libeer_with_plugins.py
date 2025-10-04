import os
import sys


def _bootstrap_paths():
    # Ensure project root on sys.path so `EEG` and `LibEER` are importable
    # Adjust this if your workspace root changes
    proj_root = "/home/ako/Project/work"
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)


def main():
    _bootstrap_paths()
    # Import adapter first so we can grab the class
    from utils.PGCN_Adapter import PGCN
    # Inject into LibEER's registry
    from LibEER.models.Models import Model as LibEERModel
    LibEERModel['PGCN'] = PGCN
    # Delegate to LibEER main
    from LibEER.utils.args import get_args_parser
    from LibEER.main import main as libeer_main

    parser = get_args_parser()
    args = parser.parse_args()
    # user can set -model PGCN now
    libeer_main(args)


if __name__ == '__main__':
    main()
