import sys
import time
from contextlib import contextmanager


@contextmanager
def profile(name: str, enabled: bool = True, output=sys.stdout):
    if not enabled:
        yield
        return

    start = time.time()
    yield

    print(f"{name} took {time.time() - start:.4f}s", file=output)
