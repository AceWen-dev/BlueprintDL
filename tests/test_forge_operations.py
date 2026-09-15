import pytest
import yaml

from dlkit.registry import METRICS
from dlkit.forge import (
    ForgeError,
    ManifestError,
    audit_project,
    export_project,
    load_project_manifest,
    scaffold_project,
)


def test_scaffold_creates_independent_auditable_project(tmp_path):
    project = scaffold_project(
        tmp_path / "projects",
        "demo-forge",
        display_name="Demo Forge",
    )

    manifest = load_project_manifest(project)
    report = audit_project(project, load_plugins=True)

    assert manifest.project.id == "demo-forge"
    assert manifest.project.package == "demo_forge"
    assert (project / "pyproject.toml").is_file()
    assert (project / "src/demo_forge/bootstrap.py").is_file()
    assert report.passed
    assert report.components == ()


def test_scaffold_refuses_invalid_identity_and_overwrite(tmp_path):
    projects = tmp_path / "projects"
    scaffold_project(projects, "safe-project")

    with pytest.raises(FileExistsError):
        scaffold_project(projects, "safe-project")
    with pytest.raises(ForgeError, match="project_id"):
        scaffold_project(projects, "../escape")
    with pytest.raises(ForgeError, match="package"):
        scaffold_project(projects, "valid", package="Invalid-Package")


def test_export_copies_only_delivery_boundary_and_refuses_overwrite(tmp_path):
    project = scaffold_project(tmp_path / "projects", "export-demo")
    undeclared = project / "local-secret.txt"
    undeclared.write_text("not for delivery", encoding="utf-8")
    cache = project / "src/export_demo/__pycache__"
    cache.mkdir()
    (cache / "module.pyc").write_bytes(b"compiled")

    exported = export_project(project, tmp_path / "deliveries")

    assert exported == tmp_path / "deliveries" / "export-demo"
    assert (exported / "project.yaml").is_file()
    assert (exported / "src/export_demo/__init__.py").is_file()
    assert (exported / ".blueprintdl-export.json").is_file()
    assert not (exported / "local-secret.txt").exists()
    assert not (exported / "src/export_demo/__pycache__").exists()

    receipt = (exported / ".blueprintdl-export.json").read_text(
        encoding="utf-8"
    )
    assert str(project) not in receipt

    with pytest.raises(FileExistsError):
        export_project(project, tmp_path / "deliveries")


def test_audit_and_export_reject_delivery_path_escape(tmp_path):
    project = scaffold_project(tmp_path / "projects", "unsafe-delivery")
    manifest_path = project / "project.yaml"
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    data["delivery"]["include"].append("../outside.txt")
    manifest_path.write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(ManifestError, match="must stay inside the project"):
        audit_project(project)
    with pytest.raises(ManifestError, match="must stay inside the project"):
        export_project(project, tmp_path / "deliveries")


def test_audit_finds_cross_project_imports(tmp_path):
    projects = tmp_path / "projects"
    first = scaffold_project(projects, "first-project")
    scaffold_project(projects, "second-project")
    source = first / "src/first_project/components/illegal.py"
    source.write_text("import second_project.models\n", encoding="utf-8")

    report = audit_project(first)
    check = next(
        item for item in report.checks if item.name == "cross-project-imports"
    )

    assert not report.passed
    assert not check.passed
    assert "second_project.models" in check.detail


def test_export_destination_must_be_outside_project(tmp_path):
    project = scaffold_project(tmp_path / "projects", "nested-export")

    with pytest.raises(ForgeError, match="outside the project"):
        export_project(project, project / "delivery")


def test_audit_tracks_owned_components_and_requires_provider(tmp_path):
    project = scaffold_project(tmp_path / "projects", "owned-project")
    components = project / "src/owned_project/components"
    (components / "metric.py").write_text(
        "from dlkit.registry import METRICS\n\n"
        "@METRICS.register(\n"
        "    name='ForgeOwnedMetricForTest',\n"
        "    provider='owned-project',\n"
        ")\n"
        "class OwnedMetric:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (components / "__init__.py").write_text(
        "def load_components():\n"
        "    from owned_project.components import metric\n"
        "    return (metric.__name__,)\n",
        encoding="utf-8",
    )

    try:
        report = audit_project(project, load_plugins=True)
        inventory = report.components

        assert report.passed
        assert len(inventory) == 1
        assert inventory[0]["provider"] == "owned-project"
        assert inventory[0]["name"] == "ForgeOwnedMetricForTest"
        assert "target" not in inventory[0]
    finally:
        METRICS.unregister("ForgeOwnedMetricForTest")


def test_audit_rejects_project_component_without_provider(tmp_path):
    project = scaffold_project(tmp_path / "projects", "unowned-project")
    components = project / "src/unowned_project/components"
    (components / "metric.py").write_text(
        "from dlkit.registry import METRICS\n\n"
        "@METRICS.register(name='ForgeUnownedMetricForTest')\n"
        "class UnownedMetric:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (components / "__init__.py").write_text(
        "def load_components():\n"
        "    from unowned_project.components import metric\n"
        "    return (metric.__name__,)\n",
        encoding="utf-8",
    )

    try:
        report = audit_project(project, load_plugins=True)
        ownership = next(
            check
            for check in report.checks
            if check.name == "component-ownership"
        )

        assert not report.passed
        assert not ownership.passed
        assert "ForgeUnownedMetricForTest" in ownership.detail
    finally:
        METRICS.unregister("ForgeUnownedMetricForTest")
