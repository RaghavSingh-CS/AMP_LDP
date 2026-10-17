"""Read-only local demo preflight. Entry imports use only the standard library."""

import argparse
import importlib
from importlib import metadata
import json
from pathlib import Path
import sys
from time import perf_counter
import tomllib

ROOT = Path(__file__).resolve().parents[1]
INSTALL = r".\.venv\Scripts\python.exe -m pip install -r requirements-interface.txt"
RESTORE = ("Restore the named file with matching SHA-256 from the approved local backup or reviewed Git revision. "
           "Original raw/processed/window inputs are local research files, not downloaded by the app. "
           "Do not rerun inference, replay, index or curve builders as a demo repair. See app/DEMO_GUIDE.md.")


def dependency_checks(root: Path):
    checks = []
    try:
        lines = (root / "requirements-interface.txt").read_text().splitlines()
        pins = [line.strip().split("==") for line in lines if line.strip() and not line.lstrip().startswith("#")]
        if len(pins) != 5 or any(len(pin) != 2 for pin in pins):
            raise ValueError("Expected the five pinned interface dependencies.")
        for name, expected in pins:
            try:
                actual = metadata.version(name)
                status = "PASS" if actual == expected else "FAIL"
                detail = f"{name}: installed {actual}; required {expected}"
            except metadata.PackageNotFoundError:
                status, detail = "FAIL", f"Missing Python dependency: {name}=={expected}"
            checks.append({"name": f"dependency: {name}", "status": status, "detail": detail,
                           "recovery": INSTALL if status == "FAIL" else "", "seconds": 0.0})
    except (OSError, ValueError) as exc:
        checks.append({"name": "dependency declaration", "status": "FAIL", "detail": str(exc),
                       "recovery": "Restore requirements-interface.txt from the reviewed interface commit.", "seconds": 0.0})
    return checks


def history_check(root: Path):
    from app.consumption_loader import load_history, verify_file
    from app.data_loader import _json, _read, _verified, _require
    from app.demo import load_demo
    catalog = _json(_read(root, "app/consumption_catalog.json"))
    for key in ("source", "test_mask", "windows"):
        verify_file(root, catalog[key])
    for key in ("index", "provenance"):
        _verified(root, catalog[key])
    examples = load_demo(root).examples
    for customer in examples.CONS_NO:
        history = load_history(customer, root)
        _require(len(history.readings) == 1035 and len(history.windows) > 0, "Presenter history is incomplete.")
    return "Original source/mask/windows/index/provenance hashes, 6,356 index members and all three selected histories verified."


def run(root: Path = ROOT):
    checks = dependency_checks(root)

    def check(name, action, recovery=RESTORE):
        started = perf_counter()
        try:
            detail = action()
            checks.append({"name": name, "status": "PASS", "detail": str(detail), "recovery": "", "seconds": round(perf_counter() - started, 4)})
        except Exception as exc:
            # A preflight diagnoses corrupt/missing dependencies and inputs, not just happy paths.
            checks.append({"name": name, "status": "FAIL", "detail": f"{type(exc).__name__}: {exc}",
                           "recovery": recovery, "seconds": round(perf_counter() - started, 4)})

    def configuration():
        config = tomllib.loads((root / ".streamlit/config.toml").read_text())
        if not (config["browser"]["gatherUsageStats"] is False
                and config["client"]["disableDataExport"] is True
                and config["client"]["toolbarMode"] == "viewer"
                and config["server"]["address"] == "127.0.0.1"):
            raise ValueError("Expected telemetry-off, viewer, disabled built-in exports and loopback demo settings.")
        return "Loopback, telemetry off, viewer toolbar and explicit aggregate-only export policy configured."

    check("local demo configuration", configuration)

    def imports():
        compile((root / "app/main.py").read_text(encoding="utf-8"), "app/main.py", "exec")
        for name in ("overview", "explorer", "inspector", "performance", "demo_guide"):
            importlib.import_module(f"app.views.{name}")
        return "Main syntax and all five view-module imports passed; no model/replay module imported."

    check("critical application imports", imports, INSTALL + "; restore named app source files if syntax/import errors persist.")
    try:
        from app.data_loader import ArtifactError, load_predictions
        from app.demo import load_demo
        from app.performance_loader import load_curves, load_performance
    except Exception as exc:
        checks.append({"name": "evidence adapters", "status": "FAIL", "detail": f"Cannot import evidence adapters: {exc}",
                       "recovery": INSTALL, "seconds": 0.0})
    else:
        check("frozen aggregates and comparators", lambda: (
            f"{load_performance(root).frozen.customers:,} customers; verified Stage 27A/B metrics and compatible comparator protocol."))

        def scores():
            result = load_predictions(root)
            if result.frame is None:
                raise ArtifactError(result.message)
            return f"{len(result.frame):,} reconstructed scores; source/provenance hashes, ordered membership and 542 positive labels verified."

        check("Stage 1B scores and provenance", scores)
        check("presenter selection provenance", lambda: f"{len(load_demo(root).examples)} real examples match reproducible evaluation-label/rank criteria.")
        check("history inputs and selected histories", lambda: history_check(root))
        started = perf_counter()
        try:
            curves = load_curves(load_performance(root), root)
            status = "PASS" if curves.provenance else "WARN"
            checks.append({"name": "optional AMP-LDP curves", "status": status, "detail": curves.message,
                           "recovery": "Aggregate comparison remains available. " + RESTORE if status == "WARN" else "",
                           "seconds": round(perf_counter() - started, 4)})
        except Exception as exc:
            checks.append({"name": "optional AMP-LDP curves", "status": "WARN", "detail": str(exc),
                           "recovery": RESTORE, "seconds": round(perf_counter() - started, 4)})
    return {"status": "FAIL" if any(c["status"] == "FAIL" for c in checks) else "PASS", "checks": checks,
            "scope": "Read-only local demo readiness; no inference, fitting, threshold selection, repair or research writes.",
            "optional_warnings": sum(c["status"] == "WARN" for c in checks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable report.")
    args = parser.parse_args()
    report = run()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"AMP DEMO PREFLIGHT: {report['status']}")
        for check in report["checks"]:
            print(f"[{check['status']}] {check['name']}: {check['detail']}")
            if check["recovery"]:
                print(f"  Recovery: {check['recovery']}")
        print(report["scope"])
    sys.exit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
