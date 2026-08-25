"""Project-local compatibility hooks loaded after activate_dev_env.ps1.

MSVC 19.44 emits UTF-8 diagnostics on this machine. PyTorch 2.12 assumes the
Windows OEM code page for C++ extension subprocesses, which can replace the
real compiler error with UnicodeDecodeError. Patch only that decode setting.
"""

from __future__ import annotations

import os

if (
    os.name == "nt"
    and os.environ.get("TECHJAM_PATCH_TORCH_CPP_EXTENSION_UTF8") == "1"
):
    try:
        import torch.utils.cpp_extension as _cpp_extension

        _cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")
    except ImportError:
        pass
