"""Read and inventory the Procurement_Desk playbook without interpreting its rules."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import warnings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PLAYBOOK_ROOT = PROJECT_ROOT / "playbook" / "playbook"
PathLike = Union[str, Path]


def locate_playbook_root(playbook_root: Optional[PathLike] = None) -> Path:
    """Return the playbook directory, using the project-relative default."""
    root = Path(playbook_root).expanduser().resolve() if playbook_root else DEFAULT_PLAYBOOK_ROOT
    if not root.is_dir():
        raise FileNotFoundError(f"Playbook directory was not found: {root}")
    return root


def _read_file(path: Path, root: Path) -> str:
    """Read one UTF-8 text file without changing its line endings."""
    relative_path = path.relative_to(root).as_posix()
    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            return file.read()
    except (OSError, UnicodeError) as error:
        raise OSError(f"Unable to read playbook file '{relative_path}': {error}") from error


def _load_files(folder: Path, root: Path) -> Dict[str, str]:
    """Read every readable file under a folder, keyed by its relative path."""
    files = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue

        relative_path = path.relative_to(root).as_posix()
        try:
            files[relative_path] = _read_file(path, root)
        except OSError as error:
            warnings.warn(str(error), RuntimeWarning, stacklevel=2)

    return files


def _load_section(section: str, playbook_root: Optional[PathLike]) -> Dict[str, str]:
    """Read all text files recursively from a required playbook section."""
    root = locate_playbook_root(playbook_root)
    section_folder = root / section
    if not section_folder.is_dir():
        raise FileNotFoundError(f"Playbook section directory was not found: {section_folder}")
    return _load_files(section_folder, root)


def read_decision_procedure(playbook_root: Optional[PathLike] = None) -> str:
    """Read policies/decision-procedure.md exactly as written."""
    root = locate_playbook_root(playbook_root)
    procedure = root / "policies" / "decision-procedure.md"
    if not procedure.is_file():
        raise FileNotFoundError(f"Decision procedure was not found: {procedure}")
    return _read_file(procedure, root)


def load_category_rules(playbook_root: Optional[PathLike] = None) -> Dict[str, str]:
    """Load category documents as raw text keyed by relative path."""
    return _load_section("categories", playbook_root)


def load_supplier_information(playbook_root: Optional[PathLike] = None) -> Dict[str, str]:
    """Load supplier documents, including any contract terms, as raw text."""
    return _load_section("suppliers", playbook_root)


def load_dated_amendments(playbook_root: Optional[PathLike] = None) -> Dict[str, str]:
    """Load amendment documents without parsing or changing their dates."""
    return _load_section("amendments", playbook_root)


def load_precedents(playbook_root: Optional[PathLike] = None) -> Dict[str, str]:
    """Load precedent documents as raw text keyed by relative path."""
    return _load_section("precedents", playbook_root)


def load_all_playbook_files(playbook_root: Optional[PathLike] = None) -> Dict[str, str]:
    """Read all readable files recursively, keyed by path relative to the playbook root."""
    root = locate_playbook_root(playbook_root)
    return _load_files(root, root)


def generate_playbook_inventory(playbook_root: Optional[PathLike] = None) -> List[str]:
    """Return all playbook file paths relative to the playbook root."""
    root = locate_playbook_root(playbook_root)
    return [
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def print_playbook_inventory(playbook_root: Optional[PathLike] = None) -> None:
    """Print the number of files in each top-level playbook section."""
    root = locate_playbook_root(playbook_root)
    inventory = generate_playbook_inventory(root)
    file_counts = {
        child.name: 0
        for child in root.iterdir()
        if child.is_dir()
    }

    for relative_path in inventory:
        section = relative_path.split("/", 1)[0] if "/" in relative_path else "root"
        file_counts[section] = file_counts.get(section, 0) + 1

    print("Playbook inventory:")
    for section in sorted(file_counts):
        print(f"  {section}: {file_counts[section]} file(s)")
    print(f"  Total: {len(inventory)} file(s)")


def export_playbook_report() -> Path:
    """Export all readable playbook files to outputs/playbook_report.txt."""
    files = load_all_playbook_files()
    report_path = PROJECT_ROOT / "outputs" / "playbook_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with report_path.open("w", encoding="utf-8", newline="") as report:
            for relative_path, content in sorted(files.items()):
                report.write(f"===== {relative_path} =====\n")
                report.write(content)
                if content and not content.endswith(("\n", "\r")):
                    report.write("\n")
                report.write("\n")
    except OSError as error:
        raise OSError(f"Unable to write playbook report '{report_path}': {error}") from error

    return report_path