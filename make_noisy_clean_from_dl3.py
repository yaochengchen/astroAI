#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits

from astroai.tools.utils import load_yaml_conf, get_irf_name, select_random_irf
from astroai.tools.ganalysis import GAnalysis


def ensure_caldb_suffix(caldb_path: str) -> str:
    if "/data/cta" not in caldb_path:
        return caldb_path.rstrip("/") + "/data/cta"
    return caldb_path


def resolve_irf(conf: dict, row: pd.Series) -> str:
    caldb_path = ensure_caldb_suffix(conf["simulation"]["caldb_path"])
    conf["simulation"]["caldb_path"] = caldb_path
    if conf["simulation"]["irf"] == "random":
        return select_random_irf(
            caldb_path=caldb_path,
            prod=conf["simulation"]["caldb"],
        )
    return get_irf_name(
        irf=row["irf"],
        caldb_path=str(Path(caldb_path) / conf["simulation"]["caldb"]),
    )


def setup_analysis(conf: dict, dl3_file: str, seed: int) -> GAnalysis:
    g = GAnalysis()
    g.set_conf(conf)
    g.set_eventfilename(dl3_file)
    try:
        g.set_reducedirfs(conf["execute"]["reducedirfdir"], seed=seed)
    except AssertionError:
        g.execute_dl3_dl4_reduction()
    return g


def make_pair(g: GAnalysis):
    dataset = g.read_dataset()
    dataset, _event_list, _gti = g.read_events(dataset)
    counts2d = dataset.counts.sum_over_axes(keepdims=False).data.astype(np.float32)
    bkg2d = dataset.background.sum_over_axes(keepdims=False).data.astype(np.float32)
    clean2d = counts2d - bkg2d
    clean2d[clean2d < 0] = 0.0
    return counts2d, clean2d


def write_fits(path: Path, data: np.ndarray):
    fits.PrimaryHDU(data=data.astype(np.float32)).writeto(path, overwrite=True)


def main():
    parser = argparse.ArgumentParser(description="Create noisy/clean FITS pairs from DL3 FITS for astroAI cleaner training.")
    parser.add_argument("-f", "--configuration", required=True, help="Path to astroAI YAML configuration.")
    parser.add_argument("-o", "--output-dir", required=True, help="Directory where noisy/ and clean/ will be created.")
    parser.add_argument("--start-seed", type=int, default=None, help="Override conf['start_seed'].")
    parser.add_argument("--samples", type=int, default=None, help="Override conf['samples'].")
    parser.add_argument("--prefix", default="crab", help="DL3 FITS filename prefix, default: crab.")
    args = parser.parse_args()

    conf = load_yaml_conf(args.configuration)
    base_dir = Path(conf["simulation"]["directory"])
    datfile = Path(conf["simulation"]["directory"]).parent / conf["simulation"]["datfile"]
    infodata = pd.read_csv(datfile, sep=" ", header=0).sort_values(by=["seed"])

    start_seed = args.start_seed if args.start_seed is not None else conf["start_seed"]
    samples = args.samples if args.samples is not None else conf["samples"]

    outdir = Path(args.output_dir)
    noisy_dir = outdir / "noisy"
    clean_dir = outdir / "clean"
    noisy_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for i in range(samples):
        seed = i + 1 + start_seed
        row_df = infodata[infodata["seed"] == seed]
        if row_df.empty:
            print(f"[WARN] seed={seed} missing in dat table, skip")
            continue
        row = row_df.iloc[0]

        dl3 = base_dir / f"{args.prefix}_{seed:05d}.fits"
        if not dl3.exists():
            print(f"[WARN] missing DL3 FITS: {dl3}, skip")
            continue

        conf["simulation"]["id"] = int(seed)
        conf["simulation"]["point_ra"] = float(row["point_ra"])
        conf["simulation"]["point_dec"] = float(row["point_dec"])
        conf["simulation"]["irf"] = resolve_irf(conf, row)

        g = setup_analysis(conf, str(dl3), int(seed))
        noisy_map, clean_map = make_pair(g)

        noisy_path = noisy_dir / f"{args.prefix}_{seed:05d}.fits"
        clean_path = clean_dir / f"{args.prefix}_{seed:05d}.fits"
        write_fits(noisy_path, noisy_map)
        write_fits(clean_path, clean_map)

        manifest_rows.append({
            "seed": int(seed),
            "dl3_file": str(dl3),
            "noisy_file": str(noisy_path),
            "clean_file": str(clean_path),
            "point_ra": float(row["point_ra"]),
            "point_dec": float(row["point_dec"]),
            "source_ra": float(row["source_ra"]) if "source_ra" in row.index else np.nan,
            "source_dec": float(row["source_dec"]) if "source_dec" in row.index else np.nan,
            "irf": str(row["irf"]),
        })
        print(f"[OK] seed={seed} -> noisy={noisy_path.name}, clean={clean_path.name}")

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = outdir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"[DONE] wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
