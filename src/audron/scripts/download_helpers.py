from __future__ import annotations

"""Download helpers for datasets referenced by the paper.

These are intentionally conservative: some sources require manual acceptance or
rate-limited access. The script writes clear instructions rather than silently
failing.
"""

import argparse
import textwrap
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description='Print dataset acquisition instructions for AUDRON.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = textwrap.dedent(
        """
        AUDRON data sources from the paper:

        1. DroneAudioDataset (GitHub)
           https://github.com/saraalemadi/DroneAudioDataset

        2. ESC-50 (GitHub or official release)
           https://github.com/karolpiczak/ESC-50

        3. Speech Commands background noise subset
           https://arxiv.org/abs/1804.03209
           TensorFlow tutorial/source includes _background_noise_ clips.

        4. DroneNoise Database (supplementary binary augmentation source)
           https://salford.figshare.com/articles/dataset/DroneNoise_Database/22133411

        Recommended local layout:

        data/raw/
          DroneAudioDataset/
          ESC-50/
          speech_commands/
          DroneNoise_Database/

        After downloading, run:
          python -m audron.scripts.prepare_real_data \
            --drone-audio-root data/raw/DroneAudioDataset \
            --drone-noise-root data/raw/DroneNoise_Database \
            --output-dir data/processed

        Note: the current environment used to build this repo did not allow live
        dataset downloads, so these steps are prepared but not executed here.
        """
    ).strip() + "\n"
    args.output.write_text(text, encoding='utf-8')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
