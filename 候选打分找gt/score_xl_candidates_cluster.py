"""Score the Human Test candidates with Model XL cosine on the cluster."""

import sys

from score_xl_candidates import main


if __name__ == "__main__":
    try:
        main(final_ranking=False)
    except KeyboardInterrupt:
        sys.exit(130)
