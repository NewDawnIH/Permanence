#!/usr/bin/env python3
"""
Generate a Markdown overview of the Travel Rule (IVMS101) data that each
jurisdiction can require from the ORIGINATOR and the BENEFICIARY.

Source: Notabene public presentation definitions
  https://pd.notabene.id/ivms101/v2/jurisdiction-thresholds.json
  -> every entry links to one presentation-definition JSON per threshold tier.

The key output per jurisdiction is the *maximum* data set: the union of every
field required in any threshold tier (and in any "pick one of" alternative).
Fields marked with a dagger (†) are only needed as one of several alternatives
in every tier where they appear, so they are not strictly mandatory by
themselves - but if you want to be safe, provide them as well.

Usage:
    python travel_rule_requirements.py
    python travel_rule_requirements.py -o overview.md --exclude-legacy

Only the standard library is required. If `pycountry` is installed, country
names are added to the output.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from urllib.error import URLError
from urllib.request import Request, urlopen

try:  # optional, only used for nicer headings
    import pycountry  # type: ignore
except ImportError:  # pragma: no cover
    pycountry = None

THRESHOLDS_URL = "https://pd.notabene.id/ivms101/v2/jurisdiction-thresholds.json"

# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #


def fetch_json(url: str, retries: int = 3, timeout: int = 30):
    """GET a URL and parse it as JSON, with a few simple retries."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "travel-rule-overview/1.0"})
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Could not fetch {url}: {last_err}")


def fetch_definition(url: str):
    """Returns (definition | None, error | None). Never raises."""
    try:
        return fetch_json(url), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


# --------------------------------------------------------------------------- #
# Turning JSON paths into readable data points
# --------------------------------------------------------------------------- #

PERSON_TYPES = {"naturalPerson": "natural", "legalPerson": "legal"}
INDEX_RE = re.compile(r"\[\d+\]")

LABELS = {
    "name": "Name",
    "nationalIdentification": "National identification",
    "geographicAddress": "Geographic address",
    "accountNumber": "Account number",
    "customerIdentification": "Customer identification",
    "countryOfResidence": "Country of residence",
}
BIRTH_LABELS = {"dateOfBirth": "Date of birth", "placeOfBirth": "Place of birth"}

Key = tuple  # (party, label)


def humanize(camel: str) -> str:
    text = re.sub(r"(?<!^)(?=[A-Z])", " ", camel).lower()
    return text[:1].upper() + text[1:]


@dataclass
class Item:
    persons: set = field(default_factory=set)  # subset of {"natural", "legal"}
    required: bool = True

    def merge(self, other: "Item") -> None:
        self.persons |= other.persons
        self.required = self.required or other.required


def parse_path(path: str):
    """'$.originator.originatorPerson[0].naturalPerson.name...' ->
    (party, label, person_type | None)."""
    tokens = [t for t in INDEX_RE.sub("", path).split(".") if t and t != "$"]
    if len(tokens) < 2:
        return None
    party, rest = tokens[0], tokens[1:]
    # skip the "originatorPerson" / "beneficiaryPerson" wrapper
    if rest and rest[0] not in PERSON_TYPES and rest[0].endswith("Person"):
        rest = rest[1:]
    person = None
    if rest and rest[0] in PERSON_TYPES:
        person = PERSON_TYPES[rest[0]]
        rest = rest[1:]
    if not rest:
        return None
    head = rest[0]
    if head == "dateAndPlaceOfBirth" and len(rest) > 1:
        label = BIRTH_LABELS.get(rest[1], humanize(rest[1]))
    else:
        label = LABELS.get(head, humanize(head))
    return party, label, person


def add_item(items: dict, key: Key, persons: set, required: bool) -> None:
    items.setdefault(key, Item(set(), False)).merge(Item(set(persons), required))


def descriptor_items(descriptor: dict) -> dict:
    """All data points listed in one input descriptor."""
    items: dict = {}
    for fld in descriptor.get("constraints", {}).get("fields", []):
        required = fld.get("predicate", "required") == "required"
        path = fld.get("path", [])
        # Either {"natural": [...], "legal": [...]} or a plain list of paths
        if isinstance(path, dict):
            groups = [(ptype, paths) for ptype, paths in path.items()]
        else:
            groups = [(None, path)]
        for declared_type, paths in groups:
            for p in paths or []:
                parsed = parse_path(p)
                if not parsed:
                    continue
                party, label, person = parsed
                ptype = declared_type if declared_type in ("natural", "legal") else person
                persons = {ptype} if ptype else {"natural", "legal"}
                add_item(items, (party, label), persons, required)
    return items


def merge_items(dicts) -> dict:
    out: dict = {}
    for d in dicts:
        for key, item in d.items():
            add_item(out, key, item.persons, item.required)
    return out


# --------------------------------------------------------------------------- #
# Interpreting one presentation definition
# --------------------------------------------------------------------------- #

PARTY_TITLES = {"originator": "Originator", "beneficiary": "Beneficiary"}


def party_title(party: str) -> str:
    return PARTY_TITLES.get(party, humanize(party))


def party_sort(party: str):
    return (0 if party == "originator" else 1 if party == "beneficiary" else 2, party)


def fmt_item(label: str, item: Item, dagger: bool = False) -> str:
    text = label
    if item.persons == {"natural"}:
        text += " (natural persons only)"
    elif item.persons == {"legal"}:
        text += " (legal persons only)"
    if not item.required:
        text += " (optional)"
    if dagger:
        text += " †"
    return text


def group_text(descriptor_item_dicts: list) -> str:
    merged = merge_items(descriptor_item_dicts)
    by_party = defaultdict(list)
    for (party, label), item in sorted(merged.items()):
        by_party[party].append(fmt_item(label, item))
    if not by_party:
        return "(nothing listed)"
    return "; ".join(
        f"**{party_title(p)}**: {', '.join(v)}"
        for p, v in sorted(by_party.items(), key=lambda kv: party_sort(kv[0]))
    )


def pick_count(req: dict) -> int:
    return req.get("count") or req.get("min") or 1


def guaranteed(req: dict, groups: dict) -> set:
    """Data points that are required whichever alternative is chosen."""
    if "from" in req:
        parts = [
            {k for k, v in d.items() if v.required} for d in groups.get(req["from"], [])
        ]
    else:
        parts = [guaranteed(n, groups) for n in req.get("from_nested", [])]
    if not parts:
        return set()
    if req.get("rule", "all") == "all" or pick_count(req) >= len(parts):
        return set().union(*parts)
    return set.intersection(*parts)


def render_lines(req: dict, groups: dict, depth: int = 0, top: bool = False) -> list:
    pad = "  " * depth
    rule = req.get("rule", "all")
    if "from" in req:
        descs = groups.get(req["from"], [])
        if rule == "pick" and len(descs) > 1:
            out = [f"{pad}- **Any {pick_count(req)} of:**"]
            out += [f"{pad}  - {group_text([d])}" for d in descs]
            return out
        return [f"{pad}- {group_text(descs)}"]
    children = req.get("from_nested", [])
    if rule == "all":
        if top:
            return [ln for c in children for ln in render_lines(c, groups, depth)]
        out = [f"{pad}- **All of:**"]
    else:
        out = [f"{pad}- **Any {pick_count(req)} of:**"]
    for c in children:
        out += render_lines(c, groups, depth + 1)
    return out


@dataclass
class Tier:
    threshold: str
    url: str
    items: dict = field(default_factory=dict)  # Key -> Item (everything listed)
    guaranteed: set = field(default_factory=set)
    lines: list = field(default_factory=list)
    error: str | None = None


def analyse_definition(tier: Tier, definition: dict) -> None:
    per_descriptor = {}
    groups: dict = defaultdict(list)
    for desc in definition.get("input_descriptors", []):
        items = descriptor_items(desc)
        per_descriptor[desc.get("id")] = items
        for g in desc.get("group", []) or []:
            groups[g].append(items)

    tier.items = merge_items(per_descriptor.values())
    reqs = definition.get("submission_requirements")
    if reqs:
        tier.lines = [ln for r in reqs for ln in render_lines(r, groups, 0, top=True)]
        tier.guaranteed = set().union(*(guaranteed(r, groups) for r in reqs))
    else:  # no rules -> every descriptor applies
        tier.lines = [f"- {group_text(list(per_descriptor.values()))}"]
        tier.guaranteed = {k for k, v in tier.items.items() if v.required}


# --------------------------------------------------------------------------- #
# Jurisdictions
# --------------------------------------------------------------------------- #


def to_decimal(value: str) -> Decimal:
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return Decimal(0)


def fmt_number(value: str) -> str:
    return f"{to_decimal(value).normalize():,f}"


@dataclass
class Jurisdiction:
    code: str
    currency: str = ""
    statuses: list = field(default_factory=list)
    page: str = ""
    tiers: dict = field(default_factory=dict)  # url -> Tier

    @property
    def legacy(self) -> bool:
        return self.code.endswith("_old")

    @property
    def status(self) -> str:
        s = ", ".join(self.statuses) or "unknown"
        return f"{s} (legacy)" if self.legacy else s

    def sorted_tiers(self) -> list:
        return sorted(self.tiers.values(), key=lambda t: to_decimal(t.threshold))

    def maximum(self) -> dict:
        return merge_items(t.items for t in self.tiers.values())

    def guaranteed_any(self) -> set:
        return set().union(*(t.guaranteed for t in self.tiers.values())) if self.tiers else set()


def country_name(code: str) -> str:
    if pycountry is None:
        return ""
    base = code.split("_")[0]
    try:
        if "-" in base:
            sub = pycountry.subdivisions.get(code=base)
            return sub.name if sub else ""
        c = pycountry.countries.get(alpha_2=base)
        return c.name if c else ""
    except (LookupError, KeyError):
        return ""


def build_jurisdictions(entries: list) -> dict:
    """Merge duplicate country codes (the source lists e.g. BR several times)."""
    result: dict = {}
    for e in entries:
        code = e["countryCode"]
        j = result.setdefault(code, Jurisdiction(code))
        j.currency = j.currency or e.get("currency", "")
        status = e.get("jurisdictionStatus")
        if status and status not in j.statuses:
            j.statuses.append(status)
        j.page = j.page or e.get("jurisdictionNotabenePage", "") or ""
        for t in e.get("thresholds", []):
            url = t["presentationDefinitionURL"]
            j.tiers.setdefault(url, Tier(threshold=str(t["threshold"]), url=url))
    return result


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #


def max_cells(j: Jurisdiction) -> dict:
    """party -> comma separated list of the maximum data points."""
    sure = j.guaranteed_any()
    by_party = defaultdict(list)
    for key, item in sorted(j.maximum().items()):
        party, label = key
        by_party[party].append(fmt_item(label, item, dagger=key not in sure))
    return {p: ", ".join(v) for p, v in by_party.items()}


def heading(j: Jurisdiction) -> str:
    name = country_name(j.code)
    return f"{j.code} - {name}" if name else j.code


def tier_title(j: Jurisdiction, t: Tier) -> str:
    return f"From {fmt_number(t.threshold)} {j.currency}".strip()


def build_markdown(jurisdictions: dict) -> str:
    js = sorted(jurisdictions.values(), key=lambda j: j.code)
    out = [
        "# Travel Rule data requirements per jurisdiction",
        "",
        f"_Generated on {date.today().isoformat()} from "
        f"[Notabene presentation definitions]({THRESHOLDS_URL})._",
        "",
        "For every jurisdiction this lists the **maximum** data that can be required: "
        "the union over all threshold tiers and all \"pick one of\" alternatives. "
        "`†` marks data that is only required as one of several alternatives "
        "(so not strictly mandatory on its own, but safe to include). "
        "Threshold tiers are shown exactly as published (amount in the jurisdiction's currency).",
        "",
        "## Summary: maximum data per jurisdiction",
        "",
        "| Jurisdiction | Status | Tiers | Originator (max) | Beneficiary (max) |",
        "|---|---|---|---|---|",
    ]
    for j in js:
        cells = max_cells(j)
        tiers = ", ".join(fmt_number(t.threshold) for t in j.sorted_tiers())
        out.append(
            f"| {heading(j)} | {j.status} | {tiers} {j.currency} | "
            f"{cells.get('originator', '-')} | {cells.get('beneficiary', '-')} |"
        )

    # ---- inverted index: data point -> jurisdictions
    index: dict = defaultdict(list)
    for j in js:
        for key in j.maximum():
            index[key].append(j.code)
    out += ["", "## Jurisdictions per data point", ""]
    for (party, label), codes in sorted(index.items(), key=lambda kv: (party_sort(kv[0][0]), kv[0][1])):
        out.append(f"- **{party_title(party)} - {label}** ({len(codes)}): {', '.join(codes)}")

    # ---- details
    out += ["", "## Details per jurisdiction", ""]
    for j in js:
        out.append(f"### {heading(j)}")
        meta = f"Status: {j.status} | Currency: {j.currency or '?'}"
        if j.page:
            meta += f" | [Notabene page]({j.page})"
        out += ["", meta, "", "**Maximum data that may be required**", ""]
        cells = max_cells(j)
        for party in sorted(cells, key=party_sort):
            out.append(f"- **{party_title(party)}**: {cells[party]}")
        out += ["", "**Per threshold tier**", ""]
        for t in j.sorted_tiers():
            out.append(f"_{tier_title(j, t)}_ (`{t.url.rsplit('/', 1)[-1]}`)")
            out.append("")
            out += t.lines if not t.error else [f"- Could not load this definition: {t.error}"]
            out.append("")
    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", default="travel_rule_requirements.md")
    ap.add_argument("--url", default=THRESHOLDS_URL, help="jurisdiction-thresholds.json location")
    ap.add_argument("--exclude-legacy", action="store_true", help="skip *_old entries")
    ap.add_argument("--workers", type=int, default=8, help="parallel downloads")
    args = ap.parse_args()

    print(f"Fetching {args.url} ...", file=sys.stderr)
    entries = fetch_json(args.url)
    jurisdictions = build_jurisdictions(entries)
    if args.exclude_legacy:
        jurisdictions = {c: j for c, j in jurisdictions.items() if not j.legacy}

    urls = sorted({u for j in jurisdictions.values() for u in j.tiers})
    print(f"Fetching {len(urls)} presentation definitions ...", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fetched = dict(zip(urls, pool.map(fetch_definition, urls)))

    failures = 0
    for j in jurisdictions.values():
        for url, tier in j.tiers.items():
            definition, err = fetched[url]
            if definition is None:
                tier.error, failures = err, failures + 1
            else:
                analyse_definition(tier, definition)

    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(build_markdown(jurisdictions))
    print(f"Wrote {args.output} ({len(jurisdictions)} jurisdictions, {failures} failed downloads)", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
