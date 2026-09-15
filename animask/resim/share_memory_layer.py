"""Make persona-scope variant packs differ ONLY in the persona layer.

The memory-layer extractions (turning points, identity memory, agentic state,
relations, world, locations, intervention) use identical prompts across
variants of the same story, so any difference between packs is sampling
noise — a confound for the oracle-vs-prefreeze contrast. This copies the
memory layer from a donor pack into a receiver pack and refolds the
receiver's profile text from its own persona + the shared memory.

Usage (from the repository root):
  python3 -m animask.resim.share_memory_layer story01 prefreeze oracle
    -> story01__oracle now shares story01__prefreeze's memory layer
"""
import json
import shutil
import sys
from pathlib import Path

from animask.resim.build_world_pack import ENGINE, fmt_profile  # noqa: E402


def main() -> None:
    sid, donor_var, recv_var = sys.argv[1], sys.argv[2], sys.argv[3]
    donor, recv = f"{sid}__{donor_var}", f"{sid}__{recv_var}"

    for rdir in sorted((ENGINE / "data" / "roles" / recv).iterdir()):
        if not (rdir / "role_info.json").exists():
            continue
        r = json.loads((rdir / "role_info.json").read_text())
        d = json.loads((ENGINE / "data" / "roles" / donor / rdir.name /
                        "role_info.json").read_text())
        r["x_memory"], r["x_agentic"] = d["x_memory"], d["x_agentic"]
        r["relation"], r["motivation"] = d["relation"], d["motivation"]
        r["activity"] = d["activity"]
        r["x_memory_donor"] = donor
        r["profile"] = fmt_profile(r["role_name"], r["x_persona"],
                                   r["x_axioms"], r["x_memory"], r["x_agentic"])
        (rdir / "role_info.json").write_text(
            json.dumps(r, indent=2, ensure_ascii=False))
        print(f"  synced memory layer: {rdir.name}")

    shutil.copy(ENGINE / "data" / "worlds" / donor / "general.json",
                ENGINE / "data" / "worlds" / recv / "general.json")
    w = json.loads((ENGINE / "data" / "worlds" / recv / "general.json").read_text())
    w["source"] = recv
    (ENGINE / "data" / "worlds" / recv / "general.json").write_text(
        json.dumps(w, indent=2, ensure_ascii=False))
    shutil.copy(ENGINE / "data" / "locations" / f"{donor}.json",
                ENGINE / "data" / "locations" / f"{recv}.json")

    pdir = ENGINE / "presets"
    dp = json.loads((pdir / f"{donor}.json").read_text())
    rp = json.loads((pdir / f"{recv}.json").read_text())
    rp["intervention"] = dp["intervention"]
    (pdir / f"{recv}.json").write_text(
        json.dumps(rp, indent=2, ensure_ascii=False))
    print(f"world/locations/intervention copied from {donor} -> {recv}")


if __name__ == "__main__":
    main()
