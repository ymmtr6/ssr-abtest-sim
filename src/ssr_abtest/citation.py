# SPDX-License-Identifier: Apache-2.0
"""Bibliographic record for the SSR paper.

``BIBTEX`` is the BibTeX exported by arXiv for 2510.08338 (fetched from
https://arxiv.org/bibtex/2510.08338). The abstract API lists version v3,
updated 27 October 2025.
"""

TITLE = (
    "LLMs Reproduce Human Purchase Intent via Semantic Similarity "
    "Elicitation of Likert Ratings"
)
AUTHORS = (
    "Benjamin F. Maier, Ulf Aslak, Luca Fiaschi, Nina Rismal, Kemble Fletcher, "
    "Christian C. Luhmann, Robbie Dow, Kli Pappas, and Thomas V. Wiecki"
)
ARXIV_ID = "2510.08338"
URL = "https://arxiv.org/abs/2510.08338"
VERSION_NOTE = "arXiv:2510.08338v3 (27 October 2025)"

# Exact export from https://arxiv.org/bibtex/2510.08338
BIBTEX = """\
@misc{maier2025llmsreproducehumanpurchase,
      title={LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings}, 
      author={Benjamin F. Maier and Ulf Aslak and Luca Fiaschi and Nina Rismal and Kemble Fletcher and Christian C. Luhmann and Robbie Dow and Kli Pappas and Thomas V. Wiecki},
      year={2025},
      eprint={2510.08338},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.08338}, 
}"""

REFERENCE_IMPLEMENTATION = "https://github.com/pymc-labs/semantic-similarity-rating"


def citation_dict() -> dict[str, str]:
    return {
        "title": TITLE,
        "authors": AUTHORS,
        "arxiv": ARXIV_ID,
        "version": VERSION_NOTE,
        "url": URL,
        "bibtex": BIBTEX,
        "reference_implementation": REFERENCE_IMPLEMENTATION,
    }
