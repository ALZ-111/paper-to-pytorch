"""
Tiny JSON results store: one file per paper, read-modify-write, always pretty-printed
so diffs stay reviewable in git.
"""

import json
import os


def load_results(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def update_results(path, updates, namespace=None):
    """Merge `updates` into the JSON at `path` and return the full document.

    With namespace=None the top-level keys are updated; with a namespace the updates
    go under that key (created if missing). Writes atomically via a temp file so an
    interrupted run cannot leave a truncated results file.
    """
    data = load_results(path)
    if namespace is None:
        data.update(updates)
    else:
        data.setdefault(namespace, {}).update(updates)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    return data
