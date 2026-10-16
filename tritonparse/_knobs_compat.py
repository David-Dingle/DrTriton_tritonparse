"""Cross-release compat for tritonparse's Triton hook points.

Triton >= 3.4 exposes tracing hook points through ``triton.knobs``:
    knobs.compilation.listener          -> the compilation listener
    knobs.autotuning.listener           -> the autotune listener
    knobs.runtime.jit_post_compile_hook -> post-JIT-compile hook
    knobs.runtime.launch_enter_hook     -> per-launch hook

Triton <= 3.3 has no ``knobs`` module. The legacy equivalents are class
attributes on the compiler/JIT objects, but only the compilation path can be
mapped cleanly:

  * compilation: there is no standalone listener, but ``CompiledKernel.__init__``
    is called once per compilation with ``(src, metadata_group)`` -- exactly the
    two inputs the knobs compilation listener consumes. We wrap that ctor and
    forward to the listener. ``maybe_trace_triton`` reads the kernel metadata
    from the JSON inside ``metadata_group`` when we pass ``metadata={}``, so the
    emitted record is faithful (python_source + metadata.name), which is all the
    downstream source-baking needs.

  * launch / jit: the legacy hooks (``CompiledKernel.launch_enter_hook``,
    ``JITFunction.compiled_hook``) use a different calling convention than the
    knobs hooks, so per-launch tracing is intentionally NOT wired on legacy
    Triton. Compilation tracing -- the part source baking relies on -- still
    works. A one-time warning is emitted so the degradation is visible.

This keeps ``structured_logging`` free of version branches: it always calls the
``set_*`` helpers below, and this module routes to knobs or the legacy path.
"""

import logging

log = logging.getLogger("tritonparse")

try:  # Triton >= 3.4
    from triton import knobs as _knobs
except Exception:  # Triton <= 3.3: no knobs module
    _knobs = None

HAS_KNOBS = _knobs is not None

# ---------------------------------------------------------------------------
# legacy compilation-listener plumbing (Triton <= 3.3)
# ---------------------------------------------------------------------------
_orig_ck_init = None
_compilation_listener = None


def _install_legacy_compilation_wrapper() -> None:
    global _orig_ck_init
    if _orig_ck_init is not None:
        return
    from triton.compiler.compiler import CompiledKernel

    _orig_ck_init = CompiledKernel.__init__

    import os as _os

    _dbg = _os.environ.get("TRITONPARSE_COMPAT_DEBUG")

    def _patched_init(self, src, metadata_group, *args, **kwargs):
        _orig_ck_init(self, src, metadata_group, *args, **kwargs)
        listener = _compilation_listener
        if listener is None:
            return
        try:
            # knobs listener signature: (src, metadata, metadata_group, times, ...)
            # metadata={} -> maybe_trace_triton loads it from the JSON in the group.
            listener(src, {}, metadata_group, None)
        except Exception as e:  # never let tracing break a compile
            if _dbg:
                log.warning("[tritonparse-compat] listener FAILED: %r", e)
            log.debug("[tritonparse] legacy compilation listener failed: %s", e)

    CompiledKernel.__init__ = _patched_init
    if _dbg:
        log.warning("[tritonparse-compat] legacy CompiledKernel.__init__ wrapper INSTALLED")


def _remove_legacy_compilation_wrapper() -> None:
    global _orig_ck_init
    if _orig_ck_init is None:
        return
    from triton.compiler.compiler import CompiledKernel

    CompiledKernel.__init__ = _orig_ck_init
    _orig_ck_init = None


# ---------------------------------------------------------------------------
# public API used by structured_logging
# ---------------------------------------------------------------------------
def set_compilation_listener(listener) -> None:
    """Install/clear the compilation listener (``None`` to clear)."""
    global _compilation_listener
    if HAS_KNOBS:
        _knobs.compilation.listener = listener
        return
    _compilation_listener = listener
    if listener is not None:
        _install_legacy_compilation_wrapper()
    else:
        _remove_legacy_compilation_wrapper()


def set_autotune_listener(listener) -> bool:
    """Install/clear the autotune listener. Returns False where Triton has no autotune listener
    (Triton <= 3.3, no knobs; Triton >= 3.7 removed knobs.autotuning.listener): a no-op there."""
    if HAS_KNOBS and hasattr(_knobs.autotuning, "listener"):
        _knobs.autotuning.listener = listener
        return True
    return False


_warned_launch = False


def _warn_launch_unavailable() -> None:
    global _warned_launch
    if not _warned_launch:
        log.warning(
            "[tritonparse] per-launch/JIT tracing is unavailable on Triton < 3.4 "
            "(no triton.knobs); compilation tracing is active, so source baking works."
        )
        _warned_launch = True


def set_jit_post_compile_hook(hook) -> None:
    if HAS_KNOBS:
        _knobs.runtime.jit_post_compile_hook = hook
        return
    if hook is not None:
        _warn_launch_unavailable()


def set_launch_enter_hook(hook) -> None:
    if HAS_KNOBS:
        _knobs.runtime.launch_enter_hook = hook
        return
    if hook is not None:
        _warn_launch_unavailable()
