from __future__ import annotations

import sys
import traceback
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


class _Raises:
    def __init__(self, exc, match=None):
        self.exc, self.match = exc, match

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"expected {self.exc.__name__} but nothing was raised")
        if not issubclass(et, self.exc):
            return False
        if self.match and self.match not in str(ev):
            raise AssertionError(f"'{self.match}' not found in '{ev}'")
        return True


class _Approx:
    def __init__(self, value, rel=1e-6, abs=1e-9):
        self.value, self.rel, self.abs = value, rel, abs

    def __eq__(self, other):
        return abs(other - self.value) <= max(self.abs, self.rel * abs(self.value))

    def __repr__(self):
        return f"approx({self.value})"


def _build_shim() -> types.ModuleType:
    m = types.ModuleType("pytest")
    m.raises = _Raises
    m.approx = lambda v, rel=1e-6, abs=1e-9: _Approx(v, rel, abs)
    m.main = lambda *a, **k: 0
    m.fixture = lambda *a, **k: (lambda f: f)
    m.mark = types.SimpleNamespace(parametrize=lambda *a, **k: (lambda f: f),
                                   skip=lambda *a, **k: (lambda f: f))
    return m


def main() -> int:
    sys.modules.setdefault("pytest", _build_shim())
    import test_pipeline  # noqa: E402

    tests = [(n, f) for n, f in vars(test_pipeline).items()
             if n.startswith("test_") and callable(f)]
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {name}: {exc}")
            traceback.print_exc()
            failed.append(name)
    print(f"\n{passed}/{len(tests)} passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    sys.exit(main())
