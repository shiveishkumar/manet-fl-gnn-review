from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

EXPECTED_HEADER = ["run", "time", "node_id", "x", "y", "speed"]
BASELINE = [(RAW / f"positions_150n_seed{i}.csv", 150, i)
            for i in range(1, 6)]
SCALABILITY = [
    (RAW / "scalability" / f"positions_{n}_run1.csv", n, 1)
    for n in (100, 200, 300, 400, 500)
]

def validate(path, expected_nodes, expected_run):
    result = {
        "file": str(path.relative_to(ROOT)),
        "expected_nodes": expected_nodes,
        "expected_run": expected_run,
        "rows": 0,
        "snapshots": 0,
        "unique_nodes": 0,
        "header_ok": False,
        "run_ids_ok": False,
        "time_range_ok": False,
        "coordinate_bounds_ok": False,
        "speed_nonnegative": False,
        "duplicate_node_time_pairs": 0,
        "pass": False,
        "errors": "",
    }
    errors = []

    if not path.is_file():
        result["errors"] = "File not found"
        return result

    try:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            result["header_ok"] = reader.fieldnames == EXPECTED_HEADER
            if not result["header_ok"]:
                errors.append(f"Unexpected header: {reader.fieldnames}")
                return result

            rows = list(reader)

        result["rows"] = len(rows)
        expected_rows = expected_nodes * 51
        if len(rows) != expected_rows:
            errors.append(f"Expected {expected_rows} rows, got {len(rows)}")

        times = set()
        nodes = set()
        nodes_by_time = {}
        pairs = set()
        duplicates = 0
        run_ok = True
        time_ok = True
        coords_ok = True
        speed_ok = True

        for row in rows:
            try:
                run = int(row["run"])
                time_s = float(row["time"])
                node = int(row["node_id"])
                x = float(row["x"])
                y = float(row["y"])
                speed = float(row["speed"])
            except (TypeError, ValueError):
                errors.append("Non-numeric or missing field")
                run_ok = time_ok = coords_ok = speed_ok = False
                continue

            if run != expected_run:
                run_ok = False
            if not (0 <= time_s <= 500):
                time_ok = False
            if not (0 <= x <= 1000 and 0 <= y <= 1000):
                coords_ok = False
            if speed < 0:
                speed_ok = False

            times.add(time_s)
            nodes.add(node)
            nodes_by_time.setdefault(time_s, set()).add(node)
            pair = (time_s, node)
            if pair in pairs:
                duplicates += 1
            pairs.add(pair)

        result["snapshots"] = len(times)
        result["unique_nodes"] = len(nodes)
        result["run_ids_ok"] = run_ok
        result["time_range_ok"] = time_ok
        result["coordinate_bounds_ok"] = coords_ok
        result["speed_nonnegative"] = speed_ok
        result["duplicate_node_time_pairs"] = duplicates

        expected_times = {float(t) for t in range(0, 501, 10)}
        if times != expected_times:
            errors.append("Snapshot times are not exactly 0, 10, ..., 500 seconds")
        expected_node_ids = set(range(expected_nodes))
        incomplete_times = [t for t, ids in nodes_by_time.items() if ids != expected_node_ids]
        if incomplete_times:
            errors.append(f"Incomplete or invalid node coverage at {len(incomplete_times)} snapshots")
        if nodes != set(range(expected_nodes)):
            errors.append("Node IDs are not exactly 0..N-1")
        if not run_ok:
            errors.append("Unexpected run ID")
        if not time_ok:
            errors.append("Time outside 0..500 seconds")
        if not coords_ok:
            errors.append("Coordinates outside 0..1000 metres")
        if not speed_ok:
            errors.append("Negative speed")
        if duplicates:
            errors.append(f"{duplicates} duplicate (time,node_id) pairs")

        result["pass"] = (
            result["header_ok"]
            and len(rows) == expected_rows
            and times == {float(t) for t in range(0, 501, 10)}
            and nodes == set(range(expected_nodes))
            and all(ids == set(range(expected_nodes)) for ids in nodes_by_time.values())
            and run_ok and time_ok and coords_ok and speed_ok
            and duplicates == 0
        )
        result["errors"] = "; ".join(errors)

    except (OSError, csv.Error) as exc:
        result["errors"] = str(exc)

    return result

results = [validate(path, n, run) for path, n, run in BASELINE + SCALABILITY]

csv_path = RESULTS / "ns3_validation.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()), lineterminator='\n')
    writer.writeheader()
    writer.writerows(results)

passed = sum(bool(r["pass"]) for r in results)
report_path = RESULTS / "ns3_validation_report.txt"
with report_path.open("w", encoding="utf-8") as f:
    f.write("NS-3 Mobility Dataset Validation\n")
    f.write("================================\n")
    f.write(f"Repository: {ROOT}\n")
    f.write(f"Files checked: {len(results)}\n")
    f.write(f"Passed: {passed}\n")
    f.write(f"Failed: {len(results) - passed}\n\n")
    for r in results:
        f.write(f"{'PASS' if r['pass'] else 'FAIL'}: {r['file']}\n")
        f.write(f"  rows={r['rows']}, snapshots={r['snapshots']}, ")
        f.write(f"nodes={r['unique_nodes']}, run={r['expected_run']}\n")
        if r["errors"]:
            f.write(f"  issues: {r['errors']}\n")
    f.write("\nThis report validates CSV structure and values; it does not\n")
    f.write("prove the files were freshly generated by rerunning NS-3.\n")

print(f"Files checked: {len(results)}")
print(f"Passed: {passed}")
print(f"Failed: {len(results) - passed}")
print(f"CSV report: {csv_path}")
print(f"Text report: {report_path}")

sys.exit(0 if passed == len(results) else 1)
