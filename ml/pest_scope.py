"""Step 1 of the pest-model task: decide the IP102 class subset on evidence.

Reads ml/_pest_class_counts.tsv (produced by ml/pest_class_counts.py) and applies the crop mapping
below, then reports which classes are in scope for each candidate subset with real image counts.

The mapping is deliberate and reviewable: every IP102 class gets (crop, tier, note).

tiers:
  named    - the class name itself names a crop the app's 60-class contract covers
  host     - a species/common name whose principal host is a covered crop (scientific names)
  poly     - polyphagous: documented on several crops, at least one of them covered
  unclear  - family/genus-level or uncertain host; not mapped without a source
  off      - principal host is a crop the app does not diagnose (beet, alfalfa, mango, ...)

IP102's own classes.txt is 1-indexed while the folders are 0-indexed; the tsv carries both, and
the scoped dataset will use *named* folders so the offset stops mattering at inference time.

App-supported crops (from android/app/src/main/assets/class_names.json, 60 classes):
  Apple, Blueberry, Cherry, Corn(maize), Cotton, Grape, Orange(citrus), Peach, Pepper,
  Potato, Raspberry, Rice, Soybean, Squash, Strawberry, Tomato, Wheat
"""

from pathlib import Path

TSV = Path(__file__).resolve().parent / "_pest_class_counts.tsv"
REPORT = Path(__file__).resolve().parent / "pest_scope_report.txt"

# idx -> (crop, tier, note)
MAPPING = {
    0: ("rice", "named", "rice leaf roller"),
    1: ("rice", "named", "rice leaf caterpillar"),
    2: ("rice", "named", "paddy stem maggot"),
    3: ("rice", "named", "asiatic rice borer"),
    4: ("rice", "named", "yellow rice borer"),
    5: ("rice", "named", "rice gall midge"),
    6: ("rice", "named", "rice stemfly"),
    7: ("rice", "named", "brown plant hopper"),
    8: ("rice", "named", "white backed plant hopper"),
    9: ("rice", "named", "small brown plant hopper"),
    10: ("rice", "named", "rice water weevil"),
    11: ("rice", "named", "rice leafhopper"),
    12: ("rice", "poly", "grain/cereal thrips - rice outbreak pest"),
    13: ("rice", "named", "rice shell pest"),
    14: ("-", "poly", "grub - soil pest of many field crops"),
    15: ("-", "poly", "mole cricket - soil pest, rice/maize/vegetables"),
    16: ("-", "poly", "wireworm - soil pest, potato/maize/wheat"),
    17: ("-", "unclear", "white margined moth - host not confirmed here"),
    18: ("-", "poly", "black cutworm - maize/wheat/cotton/tomato"),
    19: ("-", "poly", "large cutworm - maize/wheat/vegetables"),
    20: ("-", "poly", "yellow cutworm - maize/wheat/vegetables"),
    21: ("-", "poly", "red spider (tetranychid) - cotton/tomato/grape"),
    22: ("corn", "named", "corn borer"),
    23: ("-", "poly", "army worm - cereals incl. rice/wheat/maize"),
    24: ("-", "poly", "aphids (generic) - many covered crops"),
    25: ("-", "unclear", "Potosiabre vitarsis - host not confirmed here"),
    26: ("peach", "named", "peach borer"),
    27: ("wheat", "named", "english grain aphid"),
    28: ("wheat", "host", "green bug (Schizaphis graminum) - small grains"),
    29: ("wheat", "host", "bird cherry-oat aphid - cereal aphid"),
    30: ("wheat", "named", "wheat blossom midge"),
    31: ("wheat", "host", "penthaleus major (blue oat mite) - wheat/cereals"),
    32: ("-", "poly", "longlegged spider mite - cereals/maize"),
    33: ("wheat", "named", "wheat phloeothrips"),
    34: ("wheat", "named", "wheat sawfly"),
    35: ("-", "unclear", "cerodonta denticornis - cereal leaf miner, host uncertain"),
    36: ("beet", "off", "beet fly"),
    37: ("beet", "poly", "flea beetle - beet/crucifers/potato"),
    38: ("cabbage", "off", "cabbage army worm"),
    39: ("beet", "poly", "beet army worm - beet/cotton/tomato/pepper"),
    40: ("beet", "off", "beet spot flies"),
    41: ("beet", "poly", "meadow moth - beet/soybean/sunflower"),
    42: ("beet", "off", "beet weevil"),
    43: ("-", "unclear", "sericaorient alismots chulsky - host not confirmed here"),
    44: ("alfalfa", "off", "alfalfa weevil"),
    45: ("flax", "off", "flax budworm"),
    46: ("alfalfa", "off", "alfalfa plant bug"),
    47: ("-", "poly", "tarnished plant bug - cotton/strawberry/potato"),
    48: ("-", "poly", "Locustoidea (locusts) - polyphagous"),
    49: ("legume", "off", "lytta polita - bean/legume blister beetle"),
    50: ("legume", "off", "legume blister beetle"),
}

MAPPING.update({
    51: ("-", "poly", "blister beetle - polyphagous"),
    52: ("alfalfa", "off", "therioaphis maculata - spotted alfalfa aphid"),
    53: ("alfalfa", "off", "odontothrips loti"),
    54: ("-", "poly", "Thrips (generic) - tomato/pepper/cotton"),
    55: ("alfalfa", "off", "alfalfa seed chalcid"),
    56: ("cabbage", "off", "Pieris canidia"),
    57: ("cotton", "poly", "Apolygus lucorum - major cotton mirid bug"),
    58: ("-", "unclear", "Limacodidae - slug moths, hosts not confirmed here"),
    59: ("grape", "host", "Viteus vitifoliae - grape phylloxera"),
    60: ("grape", "host", "Colomerus vitis - grape erineum mite"),
    61: ("grape", "host", "Brevipoalpus lewisi - grape mite"),
    62: ("grape", "host", "oides decempunctata - grape leaf beetle"),
    63: ("-", "poly", "Polyphagotarsonemus latus - broad mite, cotton/tomato/pepper"),
    64: ("-", "poly", "Pseudococcus comstocki - apple/peach/grape/citrus"),
    65: ("grape", "host", "parathrene regalis - grape clearwing"),
    66: ("grape", "host", "Ampelophaga - grape hawk moth"),
    67: ("grape", "host", "Lycorma delicatula - spotted lanternfly (grape)"),
    68: ("grape", "host", "Xylotrechus - grape borer"),
    69: ("-", "poly", "Cicadella viridis - polyphagous leafhopper"),
    70: ("cotton", "poly", "Miridae (plant bugs) - cotton/other"),
    71: ("-", "poly", "Trialeurodes vaporariorum - greenhouse whitefly, tomato/potato"),
    72: ("grape", "host", "Erythroneura apicalis - grape leafhopper"),
    73: ("citrus", "host", "Papilio xuthus - citrus swallowtail"),
    74: ("citrus", "host", "Panonchus citri - citrus red mite"),
    75: ("olive", "off", "Phyllocoptes oleiverus - olive/citrus bud mite"),
    76: ("citrus", "host", "Icerya purchasi - cottony cushion scale"),
    77: ("citrus", "host", "Unaspis yanonensis - arrowhead scale"),
    78: ("citrus", "host", "Ceroplastes rubens - pink wax scale"),
    79: ("citrus", "host", "Chrysomphalus aonidum - Florida red scale"),
    80: ("citrus", "host", "Parlatoria zizyphus - citrus black scale"),
    81: ("citrus", "host", "Nipaecoccus vastalor - citrus mealybug"),
    82: ("citrus", "host", "Aleurocanthus spiniferus - citrus blackfly"),
    83: ("citrus", "host", "Bactrocera minax - citrus fruit fly"),
    84: ("citrus", "host", "Dacus dorsalis - oriental fruit fly"),
    85: ("citrus", "host", "Bactrocera tsuneonis - citrus fruit fly"),
    86: ("-", "poly", "Prodenia litura - cotton/tomato/vegetables"),
    87: ("-", "unclear", "Adristyrannus - host not confirmed here"),
    88: ("citrus", "host", "Phyllocnistis citrella - citrus leafminer"),
    89: ("citrus", "host", "Toxoptera citricidus - brown citrus aphid"),
    90: ("citrus", "host", "Toxoptera aurantii - black citrus aphid"),
    91: ("citrus", "host", "Aphis citricola - citrus aphid"),
    92: ("-", "poly", "Scirtothrips dorsalis - chilli/citrus/grape/cotton"),
    93: ("-", "unclear", "Dasineura sp - host not confirmed here"),
    94: ("citrus", "host", "Lawana imitata - citrus flatid planthopper"),
    95: ("citrus", "host", "Salurnis marginella - citrus flatid planthopper"),
    96: ("mango", "off", "Deporaus marginatus - mango leaf weevil"),
    97: ("mango", "off", "Chlumetia transversa - mango shoot borer"),
    98: ("mango", "off", "mango flat beak leafhopper"),
    99: ("mango", "off", "Rhytidodera bowrinii - mango stem borer"),
    100: ("mango", "off", "Sternochetus frigidus - mango seed weevil"),
    101: ("-", "poly", "Cicadellidae (leafhoppers) - polyphagous"),
})

SCOPES = {
    "S1 cereal-named (no judgment calls)": ("named",),
    "S2 S1 + host-specific species on covered crops": ("named", "host"),
    "S3 S2 + polyphagous pests of covered crops": ("named", "host", "poly"),
}


def main() -> int:
    rows = {}
    for line in TSV.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        idx, folder, cls_line, name, train, val, test, total = line.split("\t")
        rows[int(idx)] = {
            "folder": folder,
            "cls_line": int(cls_line),
            "name": name,
            "train": int(train),
            "val": int(val),
            "test": int(test),
            "total": int(total),
        }

    missing = sorted(set(rows) - set(MAPPING))
    if missing:
        raise SystemExit(f"unmapped IP102 indices: {missing}")

    out = []
    add = out.append
    add("STEP 1 - IP102 class subset for the AgroVeyra pest model")
    add("generated by ml/pest_scope.py from ml/_pest_class_counts.tsv")
    add("")
    add("Dataset: ml/data/pest/ip102/classification  (102 classes, 75,222 images, 0 corrupt)")
    add("App contract crops: 17 (Apple, Blueberry, Cherry, Corn, Cotton, Grape, Orange, Peach,")
    add("  Pepper, Potato, Raspberry, Rice, Soybean, Squash, Strawberry, Tomato, Wheat)")
    add("")
    add("=" * 118)
    add("FULL MAPPING - all 102 IP102 classes")
    add("=" * 118)
    add(f"{'idx':>3} {'cls#':>4} {'name':<31} {'crop':<8} {'tier':<8} "
        f"{'train':>5} {'val':>4} {'test':>5} {'tot':>5}  note")
    for idx in sorted(rows):
        r = rows[idx]
        crop, tier, note = MAPPING[idx]
        add(
            f"{idx:>3} {r['cls_line']:>4} {r['name'][:31]:<31} {crop:<8} {tier:<8} "
            f"{r['train']:>5} {r['val']:>4} {r['test']:>5} {r['total']:>5}  {note}"
        )

    add("")
    add("=" * 118)
    add("CANDIDATE SUBSETS")
    add("=" * 118)
    for label, tiers in SCOPES.items():
        picked = [i for i in sorted(rows) if MAPPING[i][1] in tiers]
        tr = sum(rows[i]["train"] for i in picked)
        va = sum(rows[i]["val"] for i in picked)
        te = sum(rows[i]["test"] for i in picked)
        add("")
        add(label)
        add(f"  classes={len(picked)}  train={tr}  val={va}  test={te}  total={tr + va + te}")
        by_crop = {}
        for i in picked:
            by_crop.setdefault(MAPPING[i][0], []).append(i)
        add("  by crop: " + ", ".join(
            f"{c}={len(v)}" for c, v in sorted(by_crop.items(), key=lambda kv: -len(kv[1]))))
        small = [(i, rows[i]["total"]) for i in picked if rows[i]["total"] < 200]
        if small:
            add(f"  classes under 200 images ({len(small)}): " + ", ".join(
                f"{rows[i]['name']}={t}" for i, t in sorted(small, key=lambda x: x[1])))

    excluded = [i for i in sorted(rows) if MAPPING[i][1] in ("off", "unclear")]
    add("")
    add(f"EXCLUDED under every option ({len(excluded)} classes, "
        f"{sum(rows[i]['total'] for i in excluded)} images): off-crop hosts and unmapped genera")
    add("  " + ", ".join(rows[i]["name"] for i in excluded))

    REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {REPORT.name} ({len(out)} lines)", flush=True)
    for label, tiers in SCOPES.items():
        picked = [i for i in sorted(rows) if MAPPING[i][1] in tiers]
        print(f"  {label}: {len(picked)} classes, "
              f"{sum(rows[i]['total'] for i in picked)} images", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
