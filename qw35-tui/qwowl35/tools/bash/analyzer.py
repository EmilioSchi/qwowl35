from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import os
import posixpath
import re
from typing import Any


class PatternType(Enum):
    """How a suspicious pattern should be matched against a command."""

    COMMAND = "command"
    ARGS = "args"
    SUBSTRING = "substring"
    AST = "ast"


@dataclass(frozen=True)
class SuspiciousPattern:
    """A single rule describing a suspicious shell construct."""

    command: str
    pattern_type: PatternType
    reason: str
    args: tuple[str, ...] = ()
    substring: str = ""


@dataclass
class AnalysisResult:
    """Outcome of analyzing a bash command for suspicious patterns."""

    command: str
    suspicious: bool = False
    reason: str = ""
    reasons: list[str] = field(default_factory=list)


@dataclass
class ApprovalPromptOptions:
    """Warnings and allowlist context shown when prompting for approval."""

    warnings: list[str] = field(default_factory=list)
    allowlist_info: str = ""
    review_only: bool = False


# Commands whose mere presence is considered suspicious.
_COMMAND_PATTERNS: tuple[tuple[str, str], ...] = (
    ("mkfs", "Creates a filesystem"),
    ("shred", "Securely deletes data"),
    ("dd", "Reads or overwrites disks"),
    ("mkfifo", "Creates a named pipe"),
    ("history", "Reads shell history"),
    ("nc", "Opens raw network connections"),
    ("netcat", "Opens raw network connections"),
    ("scp", "Copies files over SSH"),
    ("rsync", "Syncs files remotely"),
    ("bash", "Spawns a shell"),
    ("sh", "Spawns a shell"),
    ("zsh", "Spawns a shell"),
    ("eval", "Executes shell code"),
    ("exec", "Replaces process with command"),
    ("xargs", "Builds and runs commands"),
    ("alias", "Redefines shell commands"),
    ("arp", "Probes the local network"),
    ("users", "Reveals user identity"),
    ("netstat", "Lists network connections"),
    ("uname", "Reveals OS details"),
    ("groups", "Lists users and groups"),
    ("lsmod", "Lists loaded kernel modules"),
    ("whoami", "Reveals user identity"),
    ("id", "Reveals user and group IDs"),
    ("nmap", "Scans network hosts"),
    ("tftp", "Transfers files over TFTP"),
    ("insmod", "Loads a kernel module"),
    ("modprobe", "Loads a kernel module"),
    ("useradd", "Creates a user account"),
    ("usermod", "Modifies a user account"),
    ("crontab", "Schedules a cron job"),
    ("tcpdump", "Captures network traffic"),
    ("kill", "Terminates processes"),
    ("pkill", "Terminates processes"),
    ("sudo", "Escalates privileges"),
    ("su", "Escalates privileges"),
    ("doas", "Escalates privileges"),
)

# Commands that are suspicious only with particular arguments.
_ARGS_PATTERNS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("rm", ("-r", "-f"), "Force-deletes recursively"),
    ("rm", ("--recursive", "--force"), "Force-deletes recursively"),
    ("rm", ("-r", "--force"), "Force-deletes recursively"),
    ("rm", ("--recursive", "-f"), "Force-deletes recursively"),
    ("chmod", ("777",), "Makes files world-writable"),
    ("chmod", ("0777",), "Makes files world-writable"),
    ("chmod", ("a+rwx",), "Makes files world-writable"),
    ("chmod", ("+s",), "Sets SUID/SGID bits"),
    ("chown", (":root",), "Changes owner to root"),
    ("chown", (":0",), "Changes owner to root"),
    ("chgrp", ("root",), "Changes group to root"),
    ("chgrp", ("0",), "Changes group to root"),
    ("curl", ("-d",), "Sends request data"),
    ("curl", ("--data",), "Sends request data"),
    ("curl", ("-X", "POST"), "Sends a POST request"),
    ("curl", ("-X", "PUT"), "Sends a PUT request"),
    ("wget", ("--post",), "Sends a POST request"),
    ("wget", ("--post-data",), "Sends a POST request"),
    ("systemctl", ("stop",), "Stops a service"),
    ("systemctl", ("disable",), "Disables a service"),
    ("find", ("-name", "*password*"), "Searches for password files"),
    ("find", ("-name", "*secret*"), "Searches for secret files"),
    ("find", ("-name", "*credential*"), "Searches for credential files"),
    ("find", ("-name", "*token*"), "Searches for token files"),
    ("find", ("-name", "*.pem"), "Searches for PEM files"),
    ("find", ("-name", "*.key"), "Searches for key files"),
    ("find", ("-name", "*.p12"), "Searches for P12 files"),
    ("find", ("-name", "*id_rsa*"), "Searches for RSA keys"),
    ("find", ("-name", "*id_dsa*"), "Searches for DSA keys"),
    ("find", ("-name", "*id_ecdsa*"), "Searches for ECDSA keys"),
    ("find", ("-name", "*id_ed25519*"), "Searches for Ed25519 keys"),
    ("grep", ("-i", "password"), "Searches for passwords"),
    ("grep", ("-i", "secret"), "Searches for secrets"),
    ("grep", ("-i", "credential"), "Searches for credentials"),
    ("grep", ("-i", "token"), "Searches for tokens"),
    ("grep", ("-i", "api_key"), "Searches for API keys"),
    ("grep", ("-i", "apikey"), "Searches for API keys"),
    ("grep", ("-i", "private_key"), "Searches for private keys"),
    ("grep", ("-i", "access_key"), "Searches for access keys"),
    ("aws", ("sts", "get-caller-identity"), "Reads AWS identity"),
    ("aws", ("iam", "add-user-to-group"), "Modifies IAM users"),
    ("aws", ("iam", "attach-user-policy"), "Attaches IAM policies"),
    ("aws", ("iam", "put-user-policy"), "Creates IAM policies"),
    ("aws", ("iam", "create-access-key"), "Creates IAM access keys"),
    ("aws", ("iam", "delete-access-key"), "Deletes IAM access keys"),
    ("aws", ("iam", "list-users"), "Lists IAM users"),
    ("aws", ("iam", "list-roles"), "Lists IAM roles"),
    ("aws", ("iam", "get-user"), "Reads IAM user details"),
    ("aws", ("ec2", "describe-instances"), "Lists EC2 instances"),
    ("aws", ("ec2", "describe-key-pairs"), "Lists EC2 key pairs"),
    ("aws", ("ec2", "describe-security-groups"), "Lists EC2 security groups"),
    ("aws", ("s3", "ls"), "Lists S3 contents"),
    ("aws", ("s3", "cp"), "Copies files with S3"),
    ("aws", ("s3", "sync"), "Syncs files with S3"),
    ("setfacl", ("-R",), "Changes ACLs recursively"),
    ("setfacl", ("-m",), "Changes ACLs"),
)

# Substrings that flag a command regardless of which program runs.
_SUBSTRING_PATTERNS: tuple[tuple[str, str], ...] = (
    (":(){ :|:& };:", "Exhausts system processes"),
    (":(){ :|:& }; :", "Exhausts system processes"),
    ("/dev/sda", "Accesses raw disks"),
    ("/dev/nvme", "Accesses raw disks"),
    ("/dev/hd", "Accesses raw disks"),
    ("> /dev/", "Writes to a device"),
    (">/dev/", "Writes to a device"),
    ("/etc/shadow", "Accesses password hashes"),
    ("/etc/passwd", "Accesses account file"),
    ("/etc/sudoers", "Accesses sudo rules"),
    (".ssh/id_rsa", "Accesses SSH private key"),
    (".ssh/id_dsa", "Accesses SSH private key"),
    (".ssh/id_ecdsa", "Accesses SSH private key"),
    (".ssh/id_ed25519", "Accesses SSH private key"),
    (".ssh/config", "Accesses SSH config"),
    (".aws/credentials", "Accesses AWS credentials"),
    (".aws/config", "Accesses AWS config"),
    (".gnupg/", "Accesses GPG keys"),
    (".env", "Accesses environment secrets"),
    ("credentials.json", "Accesses credentials file"),
    ("secrets.json", "Accesses secrets file"),
    ("secrets.yaml", "Accesses secrets file"),
    ("secrets.yml", "Accesses secrets file"),
    ("-exec rm", "Runs rm via find"),
    ("-execdir rm", "Runs rm via find"),
    ("LD_PRELOAD=", "Injects a shared library"),
    ("/etc/ld.so.preload", "Modifies preload list"),
    (".bashrc", "Modifies shell startup"),
    (".bash_profile", "Modifies shell startup"),
    (".bash_history", "Accesses shell history"),
    ("~/.bash_history", "Accesses shell history"),
    (".zsh_history", "Accesses shell history"),
    ("~/.zsh_history", "Accesses shell history"),
    (".history", "Accesses shell history"),
    ("~/.history", "Accesses shell history"),
    ("chattr -i", "Removes immutable flag"),
    ("chattr +i", "Adds immutable flag"),
    ("HISTFILESIZE=0", "Disables shell history"),
    ("HISTSIZE=0", "Disables shell history"),
    ("history -c", "Clears shell history"),
    ("unset HISTFILE", "Disables history file"),
    ("service auditd stop", "Stops audit logging"),
    ("service rsyslog stop", "Stops system logging"),
    ("cat /dev/null >", "Clears file contents"),
    ("http_proxy=", "Changes HTTP proxy"),
    ("https_proxy=", "Changes HTTPS proxy"),
)


def _build_patterns() -> list[SuspiciousPattern]:
    patterns: list[SuspiciousPattern] = []
    for command, reason in _COMMAND_PATTERNS:
        patterns.append(SuspiciousPattern(command, PatternType.COMMAND, reason))
    for command, args, reason in _ARGS_PATTERNS:
        patterns.append(SuspiciousPattern(command, PatternType.ARGS, reason, args=args))
    for substring, reason in _SUBSTRING_PATTERNS:
        patterns.append(SuspiciousPattern("*", PatternType.SUBSTRING, reason, substring=substring))
    return patterns


SUSPICIOUS_PATTERNS: list[SuspiciousPattern] = _build_patterns()

_WHITESPACE = re.compile(r"\s+")
_REDIRECT = re.compile(r"(?:^|[\s;|&])((?:\d*>>?)|(?:\d*<))\s*([^\s;|&]+)")
_SAFE_PREFIX_COMMANDS = {
    "cat", "ls", "head", "tail", "less", "more", "file", "wc",
    "grep", "find", "tree", "stat", "sed",
}
_HISTORY_FILES = (".bash_history", ".zsh_history", ".history")
_SHELL_WRAPPERS = ("bash", "sh", "zsh")


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", text)


def is_numeric(text: str) -> bool:
    return len(text) > 0 and all("0" <= ch <= "9" for ch in text)


# --- tree-sitter accessors (binding-agnostic) ---


def _call(value: Any) -> Any:
    return value() if callable(value) else value


def _node_kind(node: Any) -> str:
    kind = getattr(node, "type", None)
    if kind is not None:
        return _call(kind)
    return _call(getattr(node, "kind", ""))


def _node_children(node: Any) -> list[Any]:
    children = getattr(node, "children", None)
    if children is not None and not callable(children):
        return list(children)
    if callable(children):
        return list(children())
    count = _call(getattr(node, "child_count", 0))
    return [node.child(index) for index in range(count)]


def _node_child_count(node: Any) -> int:
    children = getattr(node, "children", None)
    if children is not None and not callable(children):
        return len(children)
    return int(_call(getattr(node, "child_count", 0)))


def _node_child(node: Any, index: int) -> Any:
    children = getattr(node, "children", None)
    if children is not None and not callable(children):
        return children[index]
    return node.child(index)


def _node_text(node: Any, source: bytes) -> str:
    start = _call(getattr(node, "start_byte"))
    end = _call(getattr(node, "end_byte"))
    return source[start:end].decode("utf8", errors="replace")


def _load_parser() -> Any:
    try:
        from tree_sitter_language_pack import get_parser
    except Exception:  # pragma: no cover - optional dependency
        return None
    try:
        return get_parser("bash")
    except Exception:  # pragma: no cover - parser unavailable
        return None


def parse_bash(source: str) -> Any | None:
    """Parse a bash command into a tree-sitter tree, or None when unavailable."""
    parser = _load_parser()
    if parser is None:
        return None
    try:
        return parser.parse(source)
    except TypeError:
        return parser.parse(source.encode("utf8"))
    except Exception:  # pragma: no cover - defensive
        return None


def _root_node(tree: Any) -> Any:
    root = getattr(tree, "root_node", None)
    return _call(root)


# --- analysis ---------------------------------------------------------------


def _add_reason(result: AnalysisResult, reason: str) -> None:
    if not reason or reason in result.reasons:
        return
    result.suspicious = True
    if not result.reason:
        result.reason = reason
    result.reasons.append(reason)


def _normalize_redirect_target(target: str) -> str:
    return target.strip("\"'")


def _is_history_path(target: str) -> bool:
    for history_file in _HISTORY_FILES:
        if target == history_file or target == "~/" + history_file or target.endswith("/" + history_file):
            return True
    return False


def _collect_redirection_reasons(command: str) -> list[str]:
    normalized = normalize_whitespace(command)
    reasons: list[str] = []

    def append(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    for operator, raw_target in _REDIRECT.findall(normalized):
        target = _normalize_redirect_target(raw_target)
        if ">" in operator and _is_history_path(target):
            append("Redirects to history file")
        if target.startswith("/dev/"):
            append("Reads from a device" if "<" in operator else "Writes to a device")
    return reasons


def truncating_write_targets(command: str) -> list[str]:
    """Files the command rewrites wholesale via a truncating ``>`` redirect.

    A single ``>`` (unlike append ``>>``) replaces a file's entire contents, so
    ``cat > f``, ``echo ... > f`` and heredocs (``cat > f <<EOF``) all wholesale-
    rewrite ``f``. Append (``>>``), input (``<``) and device targets are
    excluded, as are targets carrying globs or substitutions (``$ ` * ?``) whose
    literal path is unknown. Paths are normalized so ``./f`` and ``f`` compare
    equal across calls. Order-preserving and de-duplicated.
    """
    normalized = normalize_whitespace(command)
    targets: list[str] = []
    seen: set[str] = set()
    for operator, raw_target in _REDIRECT.findall(normalized):
        if ">" not in operator or ">>" in operator:
            continue
        target = _normalize_redirect_target(raw_target)
        if not target or target.startswith("/dev/"):
            continue
        if any(ch in target for ch in "$`*?"):
            continue
        cleaned = posixpath.normpath(target)
        if cleaned in seen:
            continue
        seen.add(cleaned)
        targets.append(cleaned)
    return targets


def append_write_targets(command: str) -> list[str]:
    """Files the command appends to via ``>>`` (incremental bash authoring).

    Mirrors :func:`truncating_write_targets` but for the append operator. Used to
    flag the pattern of building or fixing an already-authored file line-by-line
    through ``echo ... >> f`` instead of the anchored edit tools — the escape
    hatch left open when only truncating ``>`` rewrites are discouraged. Input
    ``<``, device targets and globbed/substituted paths are excluded; paths are
    normalized and de-duplicated, order-preserving.
    """
    normalized = normalize_whitespace(command)
    targets: list[str] = []
    seen: set[str] = set()
    for operator, raw_target in _REDIRECT.findall(normalized):
        if ">>" not in operator:
            continue
        target = _normalize_redirect_target(raw_target)
        if not target or target.startswith("/dev/"):
            continue
        if any(ch in target for ch in "$`*?"):
            continue
        cleaned = posixpath.normpath(target)
        if cleaned in seen:
            continue
        seen.add(cleaned)
        targets.append(cleaned)
    return targets


@dataclass
class _CommandInfo:
    name: str
    args: list[str]


def _extract_string_content(node: Any, source: bytes) -> str:
    for index in range(_node_child_count(node)):
        child = _node_child(node, index)
        if _node_kind(child) == "string_content":
            return _node_text(child, source)
    return _node_text(node, source)


def _extract_raw_string_content(node: Any, source: bytes) -> str:
    content = _node_text(node, source)
    if len(content) >= 2 and content[0] == "'" and content[-1] == "'":
        return content[1:-1]
    return content


def _get_command_info(node: Any, source: bytes) -> _CommandInfo | None:
    if _node_kind(node) != "command":
        return None
    if _node_child_count(node) == 0:
        return None

    name_node = _node_child(node, 0)
    if _node_kind(name_node) != "command_name" or _node_child_count(name_node) == 0:
        return None

    word_node = _node_child(name_node, 0)
    if _node_kind(word_node) != "word":
        return None

    name = _node_text(word_node, source)
    args: list[str] = []
    for index in range(1, _node_child_count(node)):
        child = _node_child(node, index)
        kind = _node_kind(child)
        if kind in ("word", "number"):
            args.append(_node_text(child, source))
        elif kind == "string":
            args.append(_extract_string_content(child, source))
        elif kind == "raw_string":
            args.append(_extract_raw_string_content(child, source))
    return _CommandInfo(name=name, args=args)


def _has_arg(args: list[str], value: str) -> bool:
    for arg in args:
        if arg == value:
            return True
        if len(arg) >= 2 and arg[0] == "-" and len(value) >= 2 and value[0] == "-" and value[1:] in arg:
            return True
    return False


def _has_all_args(args: list[str], values: tuple[str, ...]) -> bool:
    return all(_has_arg(args, value) for value in values)


def _suspicious_reasons_for_inner(inner_command: str) -> list[str]:
    result = analyze_command(inner_command)
    return list(result.reasons)


def _check_command(node: Any, source: bytes) -> list[str]:
    info = _get_command_info(node, source)
    if info is None:
        return []

    reasons: list[str] = []

    def append(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    for pattern in SUSPICIOUS_PATTERNS:
        if pattern.pattern_type is not PatternType.COMMAND or pattern.command != info.name:
            continue
        if info.name in _SHELL_WRAPPERS:
            for index, arg in enumerate(info.args):
                if arg == "-c" and index + 1 < len(info.args):
                    return _suspicious_reasons_for_inner(" ".join(info.args[index + 1:]))
        append(pattern.reason)

    for pattern in SUSPICIOUS_PATTERNS:
        if pattern.pattern_type is not PatternType.ARGS or pattern.command != info.name:
            continue
        if _has_all_args(info.args, pattern.args):
            append(pattern.reason)

    return reasons


def _check_fork_bomb(node: Any, source: bytes) -> str:
    if _node_kind(node) != "function_definition" or _node_child_count(node) < 2:
        return ""
    name_node = _node_child(node, 0)
    if _node_kind(name_node) != "word" or _node_text(name_node, source) != ":":
        return ""
    body = _node_child(node, 1)
    body_text = _node_text(body, source)
    if "|" in body_text and "&" in body_text:
        return "Exhausts system processes"
    return ""


def _analyze_node(node: Any, source: bytes, result: AnalysisResult) -> None:
    kind = _node_kind(node)
    if kind == "command":
        for reason in _check_command(node, source):
            _add_reason(result, reason)
    elif kind == "function_definition":
        reason = _check_fork_bomb(node, source)
        if reason:
            _add_reason(result, reason)
    for index in range(_node_child_count(node)):
        _analyze_node(_node_child(node, index), source, result)


def analyze_command(command: str) -> AnalysisResult:
    """Analyze a bash command for suspicious patterns.

    Mirrors the Go reference analyzer: redirection heuristics first, then
    substring matches, then a tree-sitter AST walk for command/argument rules.
    """
    result = AnalysisResult(command=command)

    for reason in _collect_redirection_reasons(command):
        _add_reason(result, reason)

    normalized = normalize_whitespace(command)
    for pattern in SUSPICIOUS_PATTERNS:
        if pattern.pattern_type is not PatternType.SUBSTRING:
            continue
        if normalize_whitespace(pattern.substring) in normalized:
            _add_reason(result, pattern.reason)

    tree = parse_bash(command)
    if tree is None:
        return result
    _analyze_node(_root_node(tree), command.encode("utf8"), result)
    return result


def is_suspicious(command: str) -> tuple[bool, list[str]]:
    """Report whether a command is suspicious along with the matched reasons."""
    result = analyze_command(command)
    return result.suspicious, list(result.reasons)


# --- allowlist prefixes & path scoping --------------------------------------


def extract_bash_prefix(command: str) -> str:
    """Derive an allowlist prefix like ``cat:tools/`` from a safe read command.

    Returns an empty string for unsafe commands, flag-only invocations, and any
    path that uses ``..`` traversal to escape its base directory.
    """
    first_command = command.split("|", 1)[0].strip()
    fields = first_command.split()
    if len(fields) < 2:
        return ""

    base_command = fields[0]
    if base_command not in _SAFE_PREFIX_COMMANDS:
        return ""

    for arg in fields[1:]:
        if arg.startswith("-") or is_numeric(arg):
            continue
        if "/" not in arg and "\\" not in arg and not arg.startswith("."):
            continue

        arg = arg.replace("\\", "/")
        if posixpath.isabs(arg):
            return ""

        cleaned = posixpath.normpath(arg)
        if cleaned.startswith(".."):
            return ""

        if ".." in arg:
            original_base = arg.split("/", 1)[0]
            cleaned_base = cleaned.split("/", 1)[0]
            if original_base != cleaned_base:
                return ""

        is_dir = arg.endswith("/")
        directory = cleaned if is_dir else (posixpath.dirname(cleaned) or ".")
        if directory == ".":
            return f"{base_command}:./"
        return f"{base_command}:{directory}/"

    for arg in fields[1:]:
        if arg.startswith("-") or is_numeric(arg):
            continue
        return f"{base_command}:./"

    return ""


def describe_bash_allowlist(command: str) -> str:
    """Human-readable description of the allowlist scope a command would grant."""
    prefix = extract_bash_prefix(command)
    if not prefix or ":" not in prefix:
        return ""

    cmd_name, dir_path = prefix.split(":", 1)
    if dir_path != "./":
        return f"{cmd_name} in {dir_path} directory (includes subdirs)"
    return f"{cmd_name} in {dir_path} directory"


def is_command_outside_cwd(command: str) -> bool:
    """Whether any argument targets a path outside the current working directory."""
    try:
        cwd = os.path.abspath(os.getcwd())
    except OSError:
        return False

    for part in re.split(r"[|;&]", command):
        fields = part.strip().split()
        if not fields:
            continue
        for arg in fields[1:]:
            if arg.startswith("-"):
                continue
            arg = arg.strip("\"'")
            if os.path.isabs(arg):
                if not _path_is_inside(arg, cwd):
                    return True
                continue
            if arg.startswith(".."):
                resolved = os.path.normpath(os.path.join(cwd, arg))
                if not _path_is_inside(resolved, cwd):
                    return True
            if arg.startswith("~"):
                home = os.path.expanduser("~")
                if home and not _path_is_inside(home, cwd):
                    return True
    return False


def _path_is_inside(path: str, cwd: str) -> bool:
    try:
        resolved = os.path.abspath(os.path.normpath(path))
        return os.path.commonpath([cwd, resolved]) == cwd
    except ValueError:
        return False


def build_bash_approval_options(command: str) -> ApprovalPromptOptions:
    """Gather warnings and allowlist context for an approval prompt."""
    options = ApprovalPromptOptions(allowlist_info=describe_bash_allowlist(command))

    result = analyze_command(command)
    if result.suspicious:
        for reason in result.reasons:
            options.warnings.append(f"command flagged as suspicious: {reason}")

    if is_command_outside_cwd(command):
        options.warnings.append("command targets paths outside project")

    return options
