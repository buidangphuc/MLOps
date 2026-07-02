"""Entry point. Thin shim over the CLI so you can run:

    python main.py run            # full pipeline
    python main.py train --gpu 0  # one stage
    python main.py infer --text "..."

(equivalent to `python -m datascience <command>`).
"""

from datascience.cli import main

if __name__ == "__main__":
    main()
