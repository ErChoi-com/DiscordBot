"""Permanent guard against console windows flashing on Windows.

Every child process spawned by this bot must be created with a Windows
no-window flag, or Windows pops a console window for it -- exactly the bug
this test was written after (the launcher's every-minute subprocess was
spawning a visible console). The safe patterns in this codebase are:

    subprocess.run(..., **platform_support.no_window_kwargs())
    cmd, extra = platform_support.low_priority_popen_args(cmd)
    subprocess.run(cmd, **extra)
    subprocess.Popen(..., creationflags=DETACHED_PROCESS | ...)

A future contributor adding a NEW subprocess call site without one of these
has no other signal that they broke Windows -- tests pass, CI (if any) is not
Windows-only-flaky in an obvious way, and the regression only shows up as
console windows flashing on someone's desktop. This test parses every .py
file under src/ with the `ast` module (not regex, which cannot reliably
handle multi-line calls, aliasing, or track same-function variable
assignments) and fails loudly, naming file:line, whenever it finds a spawn
call that is not provably window-safe.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"

# Function names (attribute or bare) that spawn a child process.
_SUBPROCESS_FUNCS = {"run", "Popen", "call", "check_call", "check_output"}
_OS_SPAWN_PREFIXES = ("spawnl", "spawnle", "spawnlp", "spawnlpe",
                      "spawnv", "spawnve", "spawnvp", "spawnvpe")
_ASYNCIO_FUNCS = {"create_subprocess_exec", "create_subprocess_shell"}

# path/to/file.py:funcname -> reason. Keep this small; only genuinely exempt
# call sites belong here (e.g. ones that already set DETACHED_PROCESS/
# CREATE_NO_WINDOW through some means the checker cannot statically see).
ALLOWLIST: dict[str, str] = {}


class SpawnCall:
    __slots__ = ("path", "lineno", "call_desc", "safe")

    def __init__(self, path: Path, lineno: int, call_desc: str, safe: bool):
        self.path = path
        self.lineno = lineno
        self.call_desc = call_desc
        self.safe = safe

    def __str__(self) -> str:
        rel = self.path.relative_to(SRC_ROOT.parent)
        return f"{rel}:{self.lineno}: unguarded {self.call_desc} (no window-suppression kwargs)"


def _name_of(node: ast.AST) -> str | None:
    """Best-effort dotted name for a Call's func, e.g. 'subprocess.run'."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name_of(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _creationflags_is_safe(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg == "creationflags":
            try:
                src = ast.unparse(kw.value)
            except Exception:
                src = ""
            if "CREATE_NO_WINDOW" in src or "DETACHED_PROCESS" in src:
                return True
    return False


def _has_no_window_starred_kwargs(call: ast.Call, safe_star_names: set[str]) -> bool:
    for kw in call.keywords:
        # **something
        if kw.arg is None:
            try:
                src = ast.unparse(kw.value)
            except Exception:
                src = ""
            if "no_window_kwargs" in src:
                return True
            # **extra / **kwargs where `extra` was bound from
            # low_priority_popen_args earlier in the same function.
            if isinstance(kw.value, ast.Name) and kw.value.id in safe_star_names:
                return True
    return False


def _is_spawn_call(call: ast.Call, subprocess_aliases: set[str],
                    from_imports: set[str]) -> tuple[bool, str] | tuple[None, None]:
    """Returns (True, description) if `call` is a process-spawning call."""
    func = call.func
    name = _name_of(func)
    if name is None:
        return None, None

    parts = name.split(".")
    tail = parts[-1]

    if isinstance(func, ast.Attribute):
        base = parts[0]
        if base in subprocess_aliases and tail in _SUBPROCESS_FUNCS:
            return True, name
        if base == "os" and (tail == "system" or tail.startswith(_OS_SPAWN_PREFIXES) and tail in (
                "spawnl", "spawnle", "spawnlp", "spawnlpe",
                "spawnv", "spawnve", "spawnvp", "spawnvpe")):
            return True, name
        if base == "asyncio" and tail in _ASYNCIO_FUNCS:
            return True, name
    elif isinstance(func, ast.Name):
        # Bare name: only counts if imported directly from subprocess, e.g.
        # `from subprocess import run, Popen`.
        if tail in from_imports and tail in (_SUBPROCESS_FUNCS | _ASYNCIO_FUNCS):
            return True, tail
        if tail == "system" and tail in from_imports:
            return True, tail
    return None, None


def _find_spawn_calls_in_file(path: Path) -> list[SpawnCall]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    subprocess_aliases: set[str] = set()
    from_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "subprocess":
                    subprocess_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "subprocess":
                for alias in node.names:
                    from_imports.add(alias.asname or alias.name)
            elif node.module == "os":
                for alias in node.names:
                    if (alias.asname or alias.name) == "system":
                        from_imports.add("system")

    results: list[SpawnCall] = []

    def enclosing_func_body(root: ast.AST):
        """Yield (func_node) for every function/async-function definition,
        so we can scan each one's own statements for a
        low_priority_popen_args assignment that precedes a given call."""
        for node in ast.walk(root):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield node

    # Map each Call node to the nearest enclosing function (or module scope).
    func_of_call: dict[int, ast.AST] = {}
    for func in enclosing_func_body(tree):
        for node in ast.walk(func):
            if isinstance(node, ast.Call):
                func_of_call.setdefault(id(node), func)

    # For each function, collect names assigned from
    # platform_support.low_priority_popen_args(...) (tuple unpack), keyed by
    # the *kwargs-dict* variable name (second element of the tuple).
    def safe_star_names_for(func: ast.AST) -> set[str]:
        names: set[str] = set()
        for node in ast.walk(func):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                callee = _name_of(node.value.func) or ""
                if callee.endswith("low_priority_popen_args"):
                    for target in node.targets:
                        if isinstance(target, ast.Tuple) and len(target.elts) == 2:
                            second = target.elts[1]
                            if isinstance(second, ast.Name):
                                names.add(second.id)
            # A dict literal built as e.g. `kwargs = {"creationflags": ...}`
            # where the value text mentions the window/detach flags.
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                for key, val in zip(node.value.keys, node.value.values):
                    if isinstance(key, ast.Constant) and key.value == "creationflags":
                        try:
                            src = ast.unparse(val)
                        except Exception:
                            src = ""
                        if "CREATE_NO_WINDOW" in src or "DETACHED_PROCESS" in src:
                            for target in node.targets:
                                if isinstance(target, ast.Name):
                                    names.add(target.id)
            # A dict mutated afterwards: `kwargs["creationflags"] = DETACHED_PROCESS | ...`
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                        and isinstance(target.slice, ast.Constant)
                        and target.slice.value == "creationflags"):
                    try:
                        src = ast.unparse(node.value)
                    except Exception:
                        src = ""
                    if "CREATE_NO_WINDOW" in src or "DETACHED_PROCESS" in src:
                        names.add(target.value.id)
        return names

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        is_spawn, desc = _is_spawn_call(node, subprocess_aliases, from_imports)
        if not is_spawn:
            continue

        func_scope = func_of_call.get(id(node), tree)
        safe_names = safe_star_names_for(func_scope)

        safe = (
            _has_no_window_starred_kwargs(node, safe_names)
            or _creationflags_is_safe(node)
        )
        results.append(SpawnCall(path=path, lineno=node.lineno, call_desc=desc, safe=safe))

    return results


def _iter_src_files():
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def _current_funcname(path: Path, lineno: int) -> str | None:
    """Best-effort: the innermost function name enclosing `lineno`."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    best: str | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", None) or start
            if start <= lineno <= end:
                best = node.name
    return best


def _all_spawn_calls() -> list[SpawnCall]:
    calls: list[SpawnCall] = []
    for path in _iter_src_files():
        calls.extend(_find_spawn_calls_in_file(path))
    return calls


def test_checker_is_not_vacuous():
    """Sanity floor: if a refactor breaks the AST walk (new import style,
    renamed helper, etc.), this must fail loudly with zero findings rather
    than let the real test below pass vacuously."""
    calls = _all_spawn_calls()
    assert len(calls) >= 10, (
        f"expected to find at least 10 subprocess/os/asyncio spawn call sites "
        f"under src/, found {len(calls)} -- the AST walk in "
        f"test_no_unhidden_subprocess.py is likely broken (import pattern "
        f"changed, function names changed, etc.), not that spawns disappeared"
    )


def test_every_subprocess_spawn_suppresses_windows_console():
    calls = _all_spawn_calls()
    offenders = []
    for spawn in calls:
        if spawn.safe:
            continue
        rel = spawn.path.relative_to(SRC_ROOT.parent).as_posix()
        funcname = _current_funcname(spawn.path, spawn.lineno) or ""
        key = f"{rel}:{funcname}"
        if key in ALLOWLIST:
            continue
        offenders.append(spawn)

    if offenders:
        details = "\n".join(f"  - {o}" for o in offenders)
        pytest.fail(
            "Found subprocess/process-spawn call(s) under src/ with no "
            "Windows console suppression. Add "
            "**platform_support.no_window_kwargs() (or route through "
            "platform_support.low_priority_popen_args(), or set "
            "creationflags=...CREATE_NO_WINDOW/DETACHED_PROCESS explicitly) "
            "to each of:\n" + details
        )
