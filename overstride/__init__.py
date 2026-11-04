"""Overstride demo package: the offline analyzer of `docs/phase16-demo-design.md`.

This package **imports** the frozen measured pipeline and adds no new
kinematics. Anything that would change a number the sealed test measured belongs
nowhere in here -- see the design doc's section 1.
"""

__all__ = ["analyze", "preprocess"]
