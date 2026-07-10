#!/usr/bin/env python3
"""Merge the testnet extension module into the mainnet aleo wheel.

maturin builds exactly one cdylib per invocation, named by pyproject
[tool.maturin] module-name. To ship a single `aleo` wheel containing BOTH
`aleo/_aleolib_mainnet*.so` and `aleo/_aleolib_testnet*.so`, we build two wheels
(mainnet, then testnet with module-name overridden) and splice the testnet .so
into the mainnet wheel here, regenerating the RECORD so the wheel stays valid.

Usage:
    merge_testnet_so.py <mainnet_dist_dir> <testnet_dist_dir>

The mainnet wheel in <mainnet_dist_dir> is rewritten in place (repacked).
Cross-platform (pure stdlib zipfile) so it also runs on the Windows CI runner.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import sys
import zipfile


def _find_wheel(dist_dir: str) -> str:
    wheels = [f for f in os.listdir(dist_dir) if f.endswith(".whl")]
    if len(wheels) != 1:
        raise SystemExit(
            f"expected exactly one .whl in {dist_dir}, found: {wheels}"
        )
    return os.path.join(dist_dir, wheels[0])


def _find_testnet_so(wheel_path: str) -> tuple[str, bytes]:
    with zipfile.ZipFile(wheel_path) as z:
        for name in z.namelist():
            base = name.rsplit("/", 1)[-1]
            if base.startswith("_aleolib_testnet") and (
                base.endswith(".so") or base.endswith(".pyd") or base.endswith(".dylib")
            ):
                return name, z.read(name)
    raise SystemExit(f"no _aleolib_testnet extension module found in {wheel_path}")


def _record_line(arcname: str, data: bytes) -> list[str]:
    digest = hashlib.sha256(data).digest()
    b64 = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return [arcname, f"sha256={b64}", str(len(data))]


def merge(mainnet_dist: str, testnet_dist: str) -> None:
    mainnet_wheel = _find_wheel(mainnet_dist)
    testnet_wheel = _find_wheel(testnet_dist)

    so_name, so_data = _find_testnet_so(testnet_wheel)
    # Place the testnet .so alongside the mainnet one: aleo/<basename>.
    so_basename = so_name.rsplit("/", 1)[-1]
    target_arcname = f"aleo/{so_basename}"

    with zipfile.ZipFile(mainnet_wheel) as z:
        names = z.namelist()
        contents = {n: z.read(n) for n in names}
        infos = {n: z.getinfo(n) for n in names}

    if target_arcname in contents:
        print(f"note: {target_arcname} already present; overwriting")

    # Locate RECORD.
    record_name = next((n for n in names if n.endswith(".dist-info/RECORD")), None)
    if record_name is None:
        raise SystemExit("no RECORD file found in mainnet wheel")

    # Parse existing RECORD, then rebuild it with the new .so line.
    record_text = contents[record_name].decode("utf-8")
    rows = list(csv.reader(io.StringIO(record_text)))
    # Drop any prior entry for the target and the RECORD self-row (rewritten last).
    rows = [r for r in rows if r and r[0] not in (target_arcname, record_name)]
    rows.append(_record_line(target_arcname, so_data))
    # RECORD's own row carries empty hash/size per the wheel spec.
    rows.append([record_name, "", ""])

    new_record = io.StringIO()
    writer = csv.writer(new_record, lineterminator="\n")
    writer.writerows(rows)
    new_record_bytes = new_record.getvalue().encode("utf-8")

    # Repack: rewrite the wheel with the added .so and updated RECORD.
    tmp = mainnet_wheel + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            if n == record_name:
                continue  # written last
            z.writestr(infos[n], contents[n])
        z.writestr(target_arcname, so_data)
        z.writestr(record_name, new_record_bytes)

    os.replace(tmp, mainnet_wheel)
    print(f"merged {target_arcname} into {os.path.basename(mainnet_wheel)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    merge(sys.argv[1], sys.argv[2])
