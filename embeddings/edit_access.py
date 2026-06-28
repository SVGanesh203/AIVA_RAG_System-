#!/usr/bin/env python3
"""
edit_access.py

Utilities to update access.csv after a new embedding is generated.

Behavior:
- Adds a NEW column with the embedding folder name (e.g., "<slug>_<id>__<ts>")
- Sets KBase row to 1 for that column, and all other rows to 0
- If the column already exists, it will NOT duplicate it (idempotent)
- Creates a timestamped backup alongside the CSV by default

CSV assumptions:
- First column is "Designation"
- There is a row whose Designation (trimmed) equals "KBase"
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from datetime import datetime
from typing import Tuple


def update_access_csv(
    csv_path: str,
    column_name: str,
    allowed_roles: list[str] | None = None,
    grant_value: int = 1,
    default_value: int = 0,
    create_backup: bool = True,
) -> Tuple[bool, str]:
    """
    Update the access CSV by adding/ensuring a column for the given embedding.
    If allowed_roles is provided, only those designations get grant_value.
    If allowed_roles is None, we default to no one (or you can keep previous logic).
    """
    p = Path(csv_path)
    if not p.exists():
        return False, f"CSV not found: {p}"

    # Backup
    if create_backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = p.with_suffix(p.suffix + f".bak_{ts}")
        try:
            shutil.copy2(str(p), str(backup))
        except Exception as e:
            return False, f"Failed to create backup: {e}"

    # Read existing CSV
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = list(csv.reader(f))
    except Exception as e:
        return False, f"Failed to read CSV: {e}"

    if not reader:
        return False, "CSV is empty."

    header = reader[0]
    if not header or header[0].strip() != "Designation":
        return False, "Invalid CSV format: first header must be 'Designation'."

    # If column already exists, we only overwrite values for safety (idempotent)
    col_exists = column_name in header
    if not col_exists:
        header.append(column_name)

    # Build a map of column index
    col_index = {name: idx for idx, name in enumerate(header)}

    # Normalize rows to have same length as header
    rows = reader[1:]
    norm_rows = []
    for row in rows:
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        elif len(row) > len(header):
            row = row[: len(header)]
        norm_rows.append(row)

    # Write values based on allowed_roles
    target_idx = col_index[column_name]
    
    # Pre-process allowed roles for matching
    allowed_roles_norm = [r.strip().lower() for r in (allowed_roles or [])]

    for r in norm_rows:
        designation = (r[0] or "").strip()
        designation_norm = designation.lower()
        
        if designation_norm in allowed_roles_norm:
            r[target_idx] = str(grant_value)
        else:
            # Set to default_value if not an allowed role
            r[target_idx] = str(default_value)

    # Write back
    try:
        with p.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(norm_rows)
    except Exception as e:
        return False, f"Failed to write CSV: {e}"

    return True, f"Updated access CSV for column '{column_name}' with {len(allowed_roles or [])} allowed roles."


if __name__ == "__main__":
    # Tiny manual test scaffold (optional)
    import argparse

    ap = argparse.ArgumentParser(description="Update access.csv with a new embedding column.")
    ap.add_argument("--csv", required=True, help="Path to access.csv")
    ap.add_argument("--name", required=True, help="New column name (embedding folder name)")
    ap.add_argument("--kbase", default="KBase", help="Designation name for KBase row")
    ap.add_argument("--grant", type=int, default=1, help="Value to grant for KBase")
    ap.add_argument("--default", type=int, default=0, help="Default value for others")
    ap.add_argument("--no-backup", action="store_true", help="Disable creating a backup")
    args = ap.parse_args()

    ok, msg = update_access_csv(
        csv_path=args.csv,
        column_name=args.name,
        kbase_designation=args.kbase,
        grant_value=args.grant,
        default_value=args.default,
        create_backup=not args.no_backup,
    )
    print(("✅ " if ok else "❌ ") + msg)
