"""Deterministic synthetic generator for Indian crop production, 2001-2014.

Why this exists
---------------
The platform needs a dataset matching the documented 9-column schema before a
real CSV is supplied. Rather than fabricate noise, the generator encodes real
agronomic structure so that the models, the EDA charts and the zone
recommendations are all exercising something real:

* 36 states / union territories, each with its own yield and cost level
* 16 crops with authentic cultivar names and India-typical yields (q/ha)
* Kharif / Rabi / Zaid season affinity per crop
* Mandal / Village zone structure with sub-zone yield and cost variation
* Nominal cost inflation and a mild yield trend across 2001-2014
* Realistic data-quality defects: whitespace, inconsistent case, unit
  spellings, missing cells, zero/negative areas, entry-error outliers

IMPORTANT: the output is synthetic. It is a functional stand-in, not an
evidence base. Every artefact is stamped with a provenance marker so the UI can
badge it. Drop a real CSV into ``data/raw/`` and the loader prefers it instead.

Unit semantics
--------------
``unit`` describes the unit of ``production`` only. ``quantity`` is always
cultivated area in hectares, ``production`` is expressed in the stated unit.
That makes the two headline ratios interpretable:

    production_efficiency = production (quintals) / quantity (hectares)
                          -> quintals per hectare, a real yield metric
    cost_per_unit         = cost (INR) / quantity (hectares)
                          -> INR per hectare, a real cost metric
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import CROP_SEASON_AFFINITY, SEED, YEAR_MAX, YEAR_MIN

# --------------------------------------------------------------------------
# Reference agronomy: authentic cultivar names per crop
# --------------------------------------------------------------------------
CROP_VARIETIES: dict[str, list[str]] = {
    "Rice": [
        "IR-64", "Swarna", "BPT 5204", "Pusa Basmati 1121", "Basmati 1121",
        "Sarada", "Samba Mahsuri", "Birmapuri", "Mallika", "Mahima", "Jaya",
        "Indrayani", "Sona", "Dudhkala", "IR-36",
    ],
    "Wheat": [
        "HD-2967", "HD-3086", "HD-2686", "HD-2518", "PB-343", "PB-335",
        "HD-2326", "HD-2272", "LRR-201", "PB-206", "GW-322",
    ],
    "Maize": [
        "Pioneer 3396", "DKC 9108", "HQPM-1", "Vivek QPM-9", "Deccan",
        "Bio 9544", "Pearl Popcorn", "Dks 9108",
    ],
    "Cotton": [
        "RCH-659", "Bt RCH-659", "Bolltech 1412", "MCU 5", "Nanded 44",
        "Bunny Bt", "Ajeet 65", "Kaveri", "Dharwad 1",
    ],
    "Sugarcane": [
        "Co 86032", "Co 0238", "Co 86003", "VSI 8005", "SRI 26416",
        "CoM 265",
    ],
    "Groundnut": ["KTG-41", "TMV-2", "JL-24", "KRG-1", "Konar", "GNG 42"],
    "Soybean": [
        "JS-335", "JS-9305", "JS-9306", "Phule Sangam", "JS-9560", "JS-3354",
        "MACS-1188",
    ],
    "Gram": ["JG-315", "BG-159", "BG-212", "Vishwas", "Pusa Kabuli", "Kabuli BG-360"],
    "Tur": ["Vishaal", "B851", "Sona", "Maruti", "Jwala", "Bharat"],
    "Urad": ["T-8", "PU-35", "Pant Urd 31", "Barhi UR-1", "Rachna"],
    "Rapeseed": ["Pusa Mustard 25", "Pusa Mustard 26", "Sharda", "Kranti",
                 "TS-13", "RL 22 Rapeseed"],
    "Onion": [
        "Pusa Red", "Pusa White", "NHR-1", "Nasik Red", "Bellary Red",
        "Agrim Early", "Red Globe",
    ],
    "Potato": [
        "Kufri 101", "Kufri Splotch 17", "Kufri Jyoti", "Kufri Chipsona",
        "HPS-1", "Bharat", "Kufri Badshah",
    ],
    "Jute": ["TD-3", "ND-3", "JBM-1", "Tarun", "Jute Sushtra Gram"],
    "Millets": [
        "Bajra Pioneer 86", "Jowar Hybrid", "Ragi Indofinger", "Pearl Hybrid",
        "Bajra Green Gram",
    ],
}
CROP_VARIETIES["Mustard"] = CROP_VARIETIES["Rapeseed"]

# Expected output per hectare, in quintals. India-wide averages, used as the
# centre of the yield distribution.
CROP_YIELD_QUINTAL_PER_HA: dict[str, float] = {
    "Rice": 24.0,
    "Wheat": 34.0,
    "Maize": 26.0,
    "Cotton": 5.5,
    "Sugarcane": 800.0,
    "Groundnut": 11.0,
    "Soybean": 12.0,
    "Gram": 9.5,
    "Tur": 8.0,
    "Urad": 7.5,
    "Rapeseed": 12.0,
    "Mustard": 12.0,
    "Onion": 250.0,
    "Potato": 220.0,
    "Jute": 10.0,
    "Millets": 15.0,
}

# Total cost of cultivation per hectare in INR at the 2001 price level.
CROP_COST_PER_HA: dict[str, float] = {
    "Rice": 18_000.0,
    "Wheat": 15_000.0,
    "Maize": 16_000.0,
    "Cotton": 32_000.0,
    "Sugarcane": 85_000.0,
    "Groundnut": 18_000.0,
    "Soybean": 17_000.0,
    "Gram": 14_000.0,
    "Tur": 15_000.0,
    "Urad": 13_000.0,
    "Rapeseed": 16_000.0,
    "Mustard": 16_000.0,
    "Onion": 45_000.0,
    "Potato": 60_000.0,
    "Jute": 14_000.0,
    "Millets": 11_000.0,
}

# Crops that are predominantly irrigated, and therefore both higher-yielding and
# higher-cost per hectare.
IRRIGATED_SHARE = {
    "Sugarcane": 0.97,
    "Wheat": 0.93,
    "Rice": 0.92,
    "Cotton": 0.60,
    "Onion": 0.72,
    "Potato": 0.85,
    "Maize": 0.35,
    "Soybean": 0.15,
    "Groundnut": 0.20,
    "Gram": 0.25,
    "Tur": 0.10,
    "Urad": 0.15,
    "Rapeseed": 0.55,
    "Mustard": 0.55,
    "Jute": 0.20,
    "Millets": 0.15,
}

# --------------------------------------------------------------------------
# State reference data: (region, yield multiplier, cost multiplier, crops)
# --------------------------------------------------------------------------
STATE_PROFILE: dict[str, tuple[str, float, float, list[str]]] = {
    "Andhra Pradesh": ("Coastal", 1.10, 1.02, ["Rice", "Sugarcane", "Cotton", "Groundnut", "Urad"]),
    "Arunachal Pradesh": ("Himalayan", 0.72, 1.10, ["Rice", "Maize", "Millets"]),
    "Assam": ("North East", 0.78, 0.95, ["Tea", "Rice", "Jute", "Maize"]),
    "Bihar": ("Eastern", 0.88, 0.86, ["Rice", "Wheat", "Maize", "Gram", "Urad"]),
    "Chhattisgarh": ("Central", 0.83, 0.82, ["Rice", "Wheat", "Cotton", "Millets"]),
    "Goa": ("Coastal", 0.85, 1.18, ["Rice", "Sugarcane", "Cashew"]),
    "Gujarat": ("Western", 1.05, 1.00, ["Cotton", "Groundnut", "Wheat", "Rapeseed"]),
    "Haryana": ("Northern", 1.18, 1.15, ["Wheat", "Rice", "Cotton", "Mustard"]),
    "Himachal Pradesh": ("Himalayan", 0.90, 1.20, ["Apple", "Wheat", "Maize", "Rapeseed"]),
    "Jammu and Kashmir": ("Himalayan", 0.85, 1.15, ["Rice", "Wheat", "Maize", "Millets"]),
    "Jharkhand": ("Eastern", 0.75, 0.84, ["Rice", "Maize", "Millets", "Tur"]),
    "Karnataka": ("Southern", 1.06, 1.04, ["Sugarcane", "Cotton", "Millets", "Soybean", "Tur"]),
    "Kerala": ("Coastal", 1.12, 1.30, ["Rice", "Coconut", "Spices", "Cashew"]),
    "Madhya Pradesh": ("Central", 1.02, 0.90, ["Wheat", "Soybean", "Cotton", "Gram", "Urad", "Sugarcane"]),
    "Maharashtra": ("Western", 0.95, 1.06, ["Cotton", "Soybean", "Wheat", "Urad", "Sugarcane", "Onion"]),
    "Manipur": ("North East", 0.70, 1.05, ["Rice", "Maize", "Millets"]),
    "Meghalaya": ("North East", 0.68, 1.08, ["Rice", "Maize", "Millets"]),
    "Mizoram": ("North East", 0.72, 1.12, ["Rice", "Maize", "Millets"]),
    "Nagaland": ("North East", 0.70, 1.10, ["Rice", "Maize", "Millets"]),
    "Odisha": ("Eastern", 0.90, 0.88, ["Rice", "Groundnut", "Millets", "Tur"]),
    "Punjab": ("Northern", 1.25, 1.18, ["Wheat", "Rice", "Cotton", "Mustard"]),
    "Rajasthan": ("Western", 0.86, 1.02, ["Wheat", "Mustard", "Bajra", "Urad", "Groundnut"]),
    "Sikkim": ("Himalayan", 0.75, 1.25, ["Maize", "Rice", "Cardamom"]),
    "Tamil Nadu": ("Southern", 1.14, 1.08, ["Sugarcane", "Rice", "Millets", "Groundnut", "Tur"]),
    "Telangana": ("Deccan", 1.05, 1.02, ["Cotton", "Rice", "Soybean", "Millets"]),
    "Tripura": ("North East", 0.76, 0.96, ["Rice", "Maize", "Jute", "Millets"]),
    "Uttar Pradesh": ("Northern", 1.12, 0.92, ["Wheat", "Rice", "Urad", "Sugarcane", "Potato", "Gram"]),
    "Uttarakhand": ("Himalayan", 0.94, 1.08, ["Wheat", "Rice", "Sugarcane", "Potato"]),
    "West Bengal": ("Eastern", 1.15, 1.05, ["Rice", "Jute", "Potato", "Maize", "Tur"]),
    "Andaman and Nicobar Islands": ("Coastal", 0.70, 1.30, ["Coconut", "Spices", "Rice"]),
    "Chandigarh": ("Northern", 1.00, 1.25, ["Wheat", "Rice"]),
    "Dadra and Nagar Haveli and Daman and Diu": ("Western", 0.85, 1.15, ["Rice", "Groundnut"]),
    "Delhi": ("Northern", 1.05, 1.35, ["Wheat", "Rice", "Onion", "Potato"]),
    "Jammu": ("Himalayan", 0.88, 1.12, ["Rice", "Wheat", "Maize"]),
    "Ladakh": ("Himalayan", 0.55, 1.30, ["Barley", "Potato"]),
    "Lakshadweep": ("Coastal", 0.65, 1.28, ["Coconut", "Fish"]),
    "Puducherry": ("Coastal", 1.10, 1.12, ["Rice", "Sugarcane", "Groundnut"]),
}

# Crops referenced above but not part of the modelled catalogue are mapped onto
# a close analogue so every state has full coverage.
CROP_ALIAS = {
    "Tea": "Millets",
    "Cashew": "Millets",
    "Coconut": "Millets",
    "Spices": "Tur",
    "Apple": "Millets",
    "Cardamom": "Tur",
    "Bajra": "Millets",
    "Barley": "Wheat",
    "Fish": "Millets",
    "Oil Palm": "Millets",
}

REAL_MANDALS: dict[str, list[str]] = {
    "Northern": ["Ludhiana", "Amritsar", "Sangrur", "Hisar", "Rohtak", "Panipat",
                 "Karnal", "Shahjahanpur", "Bareilly", "Muzaffarnagar", "Meerut",
                 "Aligarh", "Ghaziabad"],
    "Eastern": ["Nadia", "Murshidabad", "Malda", "Dakshin Dinajpur", "Patna",
                "Gaya", "Nalanda", "Palamu", "Hazaribagh", "Cuttack", "Puri",
                "Baleshwar", "Hooghly"],
    "Western": ["Nashik", "Pune", "Solapur", "Ahmedabad", "Rajkot", "Bardoli",
                "Jaipur", "Jodhpur", "Bikaner", "Jalgaon", "Amravati", "Akola"],
    "Central": ["Indore", "Bhopal", "Dhar", "Jabalpur", "Rewa", "Satna",
                "Raipur", "Bilaspur", "Durg", "Kota", "Ujjain"],
    "Southern": ["Krishnagiri", "Thoothukudi", "Erode", "Tirunelveli", "Belgaum",
                 "Dharwad", "Mysore", "Mandya", "Salem", "Villupuram", "Kochi",
                 "Thrissur", "Warangal"],
    "Coastal": ["East Godavari", "West Godavari", "Guntur", "Nellore", "Visakhapatnam",
                "Kozhikode", "Thrissur", "Ernakulam", "Dakshina Kannada", "Puducherry"],
    "Himalayan": ["Shimla", "Kullu", "Mandi", "Kangra", "Srinagar", "Anantnag",
                  "Dehradun", "Haridwar", "Chamoli", "Kupwara", "Jammu"],
    "North East": ["Nagaon", "Darrang", "Imphal East", "Ri Bhoi", "East Khasi Hills",
                   "Aizawl", "Wokha", "Dimapur", "Agartala", "North Tripura",
                   "Tura", "Golpara"],
    "Deccan": ["Adilabad", "Nalgonda", "Warangal", "Karimnagar", "Khammam",
               "Mahabubnagar", "Medak"],
}

MANDAL_STATE: dict[str, str] = {
    # Northern
    "Ludhiana": "Punjab", "Amritsar": "Punjab", "Sangrur": "Punjab",
    "Hisar": "Haryana", "Rohtak": "Haryana", "Panipat": "Haryana",
    "Karnal": "Haryana",
    "Shahjahanpur": "Uttar Pradesh", "Bareilly": "Uttar Pradesh",
    "Muzaffarnagar": "Uttar Pradesh", "Meerut": "Uttar Pradesh",
    "Aligarh": "Uttar Pradesh", "Ghaziabad": "Uttar Pradesh",
    # Eastern
    "Nadia": "West Bengal", "Murshidabad": "West Bengal",
    "Malda": "West Bengal", "Dakshin Dinajpur": "West Bengal",
    "Baleshwar": "West Bengal", "Hooghly": "West Bengal",
    "Patna": "Bihar", "Gaya": "Bihar", "Nalanda": "Bihar",
    "Palamu": "Jharkhand", "Hazaribagh": "Jharkhand",
    "Cuttack": "Odisha", "Puri": "Odisha",
    # Western
    "Nashik": "Maharashtra", "Pune": "Maharashtra",
    "Solapur": "Maharashtra", "Jalgaon": "Maharashtra",
    "Amravati": "Maharashtra", "Akola": "Maharashtra",
    "Ahmedabad": "Gujarat", "Rajkot": "Gujarat", "Bardoli": "Gujarat",
    "Jaipur": "Rajasthan", "Jodhpur": "Rajasthan", "Bikaner": "Rajasthan",
    # Central
    "Indore": "Madhya Pradesh", "Bhopal": "Madhya Pradesh",
    "Dhar": "Madhya Pradesh", "Jabalpur": "Madhya Pradesh",
    "Rewa": "Madhya Pradesh", "Satna": "Madhya Pradesh",
    "Ujjain": "Madhya Pradesh",
    "Raipur": "Chhattisgarh", "Bilaspur": "Chhattisgarh",
    "Durg": "Chhattisgarh", "Kota": "Rajasthan",
    # Southern
    "Krishnagiri": "Tamil Nadu", "Thoothukudi": "Tamil Nadu",
    "Erode": "Tamil Nadu", "Tirunelveli": "Tamil Nadu",
    "Salem": "Tamil Nadu", "Villupuram": "Tamil Nadu",
    "Belgaum": "Karnataka", "Dharwad": "Karnataka",
    "Mysore": "Karnataka", "Mandya": "Karnataka",
    "Kochi": "Kerala", "Thrissur": "Kerala",
    # Coastal
    "East Godavari": "Andhra Pradesh", "West Godavari": "Andhra Pradesh",
    "Guntur": "Andhra Pradesh", "Nellore": "Andhra Pradesh",
    "Visakhapatnam": "Andhra Pradesh",
    "Kozhikode": "Kerala", "Ernakulam": "Kerala",
    "Dakshina Kannada": "Karnataka", "Puducherry": "Puducherry",
    # Himalayan
    "Shimla": "Himachal Pradesh", "Kullu": "Himachal Pradesh",
    "Mandi": "Himachal Pradesh", "Kangra": "Himachal Pradesh",
    "Srinagar": "Jammu and Kashmir", "Anantnag": "Jammu and Kashmir",
    "Kupwara": "Jammu and Kashmir", "Jammu": "Jammu and Kashmir",
    "Dehradun": "Uttarakhand", "Haridwar": "Uttarakhand",
    "Chamoli": "Uttarakhand",
    # North East
    "Nagaon": "Assam", "Darrang": "Assam", "Golpara": "Assam",
    "Ri Bhoi": "Meghalaya", "East Khasi Hills": "Meghalaya",
    "Tura": "Meghalaya", "Imphal East": "Manipur",
    "Aizawl": "Mizoram", "Wokha": "Nagaland", "Dimapur": "Nagaland",
    "Agartala": "Tripura", "North Tripura": "Tripura",
    # Deccan
    "Adilabad": "Telangana", "Nalgonda": "Telangana",
    "Karimnagar": "Telangana", "Khammam": "Telangana",
    "Mahabubnagar": "Telangana", "Medak": "Telangana",
    "Warangal": "Telangana",
}

#: Every mandal in :data:`REAL_MANDALS` must be attributable to exactly one state,
#: otherwise zone names would contradict the state they are listed under.
assert set(MANDAL_STATE) == {
    m for mandals in REAL_MANDALS.values() for m in mandals
}, "MANDAL_STATE and REAL_MANDALS disagree on the mandal set"


def validate_crop_catalogue() -> None:
    """Every canonical crop must be plantable somewhere, and vice versa.

    A crop with a price and a variety list but no state growing it would show
    up in the dashboard's dropdown and then fail to recommend or predict
    anything, which is worse than it not existing at all.
    """
    from src.schema import CROP_ALIASES

    canonical = set(CROP_ALIASES.values())
    plantable = set(CROP_VARIETIES) | set(CROP_SEASON_AFFINITY)
    in_states = {
        CROP_ALIAS.get(crop, crop)
        for *_, pool in STATE_PROFILE.values()
        for crop in pool
    }

    missing_state = sorted(canonical - in_states)
    if missing_state:
        raise AssertionError(
            f"crops declared but never grown in any state: {missing_state}"
        )
    unknown = sorted(in_states - plantable)
    if unknown:
        raise AssertionError(
            f"state crop pools reference unknown crops: {unknown}"
        )
    no_affinity = sorted(set(CROP_SEASON_AFFINITY) - canonical)
    if no_affinity:
        raise AssertionError(f"season affinity for undeclared crops: {no_affinity}")


def mandals_for_state(state: str) -> list[str]:
    """Mandals genuinely located in ``state``, in stable alphabetical order."""
    return sorted(m for m, home in MANDAL_STATE.items() if home == state)


VILLAGE_STEMS = [
    "Rampur", "Shivpur", "Belpur", "Kishanganj", "Narsinghpur", "Devgarh",
    "Amraiwadi", "Chandpura", "Gulabpur", "Sultanpur", "Bhagatpur", "Mohanpur",
    "Ratanpur", "Salempur", "Dhanora", "Kakori", "Bishnupur", "Madhopur",
    "Lakhanpur", "Nayanagar", "Tilakpur", "Barwala",
]
VILLAGE_SUFFIXES = ["", " Khurd", " Kalan", " Buzurg", " Nayabad", " Tola"]

CROPS_WITH_HIGH_VALUE_OUTPUT = {
    "Cotton", "Gram", "Tur", "Urad", "Groundnut", "Soybean", "Rapeseed", "Mustard",
}

# --------------------------------------------------------------------------
# Growth factors
# --------------------------------------------------------------------------
# Nominal input-cost inflation across the window (indexed to 2001 = 1.0).
COST_INFLATION_BY_YEAR = {
    2001: 1.00, 2002: 1.04, 2003: 1.07, 2004: 1.12, 2005: 1.16, 2006: 1.21,
    2007: 1.28, 2008: 1.39, 2009: 1.44, 2010: 1.52, 2011: 1.66, 2012: 1.79,
    2013: 1.95, 2014: 2.06,
}

# Broad yield improvement (irrigation, seed replacement, fertiliser adoption).
YIELD_TREND_BY_YEAR = {
    2001: 0.90, 2002: 0.92, 2003: 0.93, 2004: 0.95, 2005: 0.97, 2006: 0.99,
    2007: 1.01, 2008: 1.03, 2009: 1.04, 2010: 1.06, 2011: 1.07, 2012: 1.09,
    2013: 1.10, 2014: 1.12,
}

# Area distribution (log-normal): the long tail of very large holdings and the
# mass of small and marginal farms is what makes `Quantity` heavy-tailed.
AREA_MEDIAN_HA = 2.4
AREA_SIGMA = 1.15

NOISE_SIGMA_YIELD = 0.17
NOISE_SIGMA_COST = 0.13


def _build_zone_names(rng: np.random.Generator) -> dict[str, list[str]]:
    """Stable mandal -> village structure.

    A mandal is only ever attached to the state it actually lies in. Handing
    Punjab a Haryana district would make every zone recommendation visibly
    wrong, so each state's pool comes from :func:`mandals_for_state`. States
    with no districts in the table contribute no zones at all and fall back to
    the state-wide ``State - X`` label, which is how a single-district or
    city-only state would genuinely be reported.
    """
    zones: dict[str, list[str]] = {}

    for state, *_rest in STATE_PROFILE.items():
        own = mandals_for_state(str(state))
        if not own:
            continue

        upper = min(8, len(own))
        n = int(rng.integers(1, upper + 1))
        chosen = rng.choice(own, size=n, replace=False)

        villages: list[str] = []
        # ruff: noqa: B007 - `mandal` IS used below; B007 is a false positive here
        for mandal in chosen:
            k = int(rng.integers(2, 5))
            for _ in range(k):
                stem = VILLAGE_STEMS[int(rng.integers(0, len(VILLAGE_STEMS)))]
                suffix = VILLAGE_SUFFIXES[
                    int(rng.integers(0, len(VILLAGE_SUFFIXES)))
                ]
                villages.append(f"{stem}{suffix}")
        zones[str(mandal)] = villages
    return zones


def _zone_label(
    rng: np.random.Generator,
    state: str,
    mandal: str,
    village: str,
    has_mandals: bool,
) -> str:
    """Pick the most specific zone label the state can actually support.

    States with no districts in :data:`MANDAL_STATE` can only be reported at
    state level, so they never receive a mandal or village name borrowed from
    somewhere else.
    """
    if not has_mandals or rng.random() < 0.22:
        return f"State - {state}"
    if rng.random() < 0.68:
        return f"Mandal - {mandal}"
    return f"Village - {village}"


def generate_synthetic_dataset(
    n_rows: int = 60_000,
    seed: int = SEED,
    dirty: bool = True,
) -> pd.DataFrame:
    """Generate a schema-conformant synthetic crop production dataset.

    Parameters
    ----------
    n_rows:
        Approximate number of observations to produce.
    seed:
        Seed for the numpy Generator; identical seeds give identical output.
    dirty:
        When True, inject the data-quality defects a real extract would carry
        (whitespace, mixed case, inconsistent unit spellings, missing cells,
        zero/negative areas and entry-error outliers). The loader is expected to
        clean all of it.
    """
    validate_crop_catalogue()
    rng = np.random.default_rng(seed)

    states = list(STATE_PROFILE)
    # Sampling weights shaped by how much farmland each region actually has.
    state_weights = np.array(
        [
            {"Northern": 3.1, "Western": 3.0, "Southern": 2.4, "Eastern": 2.2,
             "Central": 2.0, "Coastal": 1.3, "Himalayan": 0.8,
             "North East": 0.6, "Deccan": 0.9}.get(STATE_PROFILE[s][0], 1.0)
            for s in states
        ],
        dtype=float,
    )
    state_weights = state_weights / state_weights.sum()

    zone_names = _build_zone_names(rng)
    mandals_by_state: dict[str, list[str]] = {
        str(state): [m for m in mandals_for_state(str(state)) if m in zone_names]
        for state in STATE_PROFILE
    }

    all_mandals = sorted(zone_names)
    mandal_rng = {m: rng.normal(0.0, 0.13) for m in all_mandals}

    years = np.arange(YEAR_MIN, YEAR_MAX + 1)
    year_weights = np.array([1.0 + 0.012 * i for i in range(len(years))])
    year_weights /= year_weights.sum()

    records: list[dict] = []

    while len(records) < n_rows:
        remaining = n_rows - len(records)
        batch = int(min(remaining, 400_000))

        s_idx = rng.choice(len(states), size=batch, p=state_weights)
        state_arr = np.array(states, dtype=object)[s_idx]

        crops = np.empty(batch, dtype=object)
        seasons = np.empty(batch, dtype=object)
        varieties = np.empty(batch, dtype=object)
        mandals = np.empty(batch, dtype=object)
        villages = np.empty(batch, dtype=object)
        # Per-row flag, not a loop variable: whether a state has districts is a
        # property of the row, and reading the batch loop's last value when
        # building records would give every state in the batch one answer.
        has_mandals_arr = np.zeros(batch, dtype=bool)
        units = np.empty(batch, dtype=object)
        state_yield_mult = np.empty(batch, dtype=float)
        state_cost_mult = np.empty(batch, dtype=float)

        for i, state in enumerate(state_arr):
            _region, y_mult, c_mult, crop_pool = STATE_PROFILE[state]
            crop_pool = [CROP_ALIAS.get(c, c) for c in crop_pool]
            crop_pool = [c for c in crop_pool if c in CROP_SEASON_AFFINITY] or ["Rice"]
            crop = crop_pool[int(rng.integers(0, len(crop_pool)))]

            affinity = CROP_SEASON_AFFINITY[crop]
            if len(affinity) == 1:
                season = affinity[0]
            else:
                season = affinity[int(rng.integers(0, len(affinity)))]

            var_pool = CROP_VARIETIES.get(crop, ["Local"])
            variety = var_pool[int(rng.integers(0, len(var_pool)))]

            mandal_pool = mandals_by_state[state]
            if mandal_pool:
                mandal = mandal_pool[int(rng.integers(0, len(mandal_pool)))]
                village_pool = zone_names[mandal]
                village = village_pool[int(rng.integers(0, len(village_pool)))]
            else:
                mandal = ""
                village = ""

            crops[i] = crop
            seasons[i] = season
            varieties[i] = variety
            mandals[i] = mandal
            villages[i] = village
            has_mandals_arr[i] = bool(mandal_pool)
            units[i] = "Quintals" if rng.random() < 0.78 else "Tons"
            state_yield_mult[i] = y_mult
            state_cost_mult[i] = c_mult

        years_arr = rng.choice(years, size=batch, p=year_weights)

        # --- Cultivated area (hectares) -------------------------------------
        area = np.exp(
            np.log(AREA_MEDIAN_HA) + rng.normal(0.0, AREA_SIGMA, size=batch)
        )
        area = np.clip(area, 0.02, 5_000.0)

        inflation = np.array([COST_INFLATION_BY_YEAR[int(y)] for y in years_arr])
        trend = np.array([YIELD_TREND_BY_YEAR[int(y)] for y in years_arr])

        irrigated_rate = np.array(
            [IRRIGATED_SHARE.get(str(c), 0.4) for c in crops], dtype=float
        )
        is_irrigated = rng.random(batch) < irrigated_rate

        zone_yield = np.array([mandal_rng.get(str(m), 0.0) for m in mandals])
        zone_cost = np.array([mandal_rng.get(str(m), 0.0) * 0.7 for m in mandals])

        base_yield = np.array(
            [CROP_YIELD_QUINTAL_PER_HA.get(str(c), 15.0) for c in crops], dtype=float
        )
        base_cost = np.array(
            [CROP_COST_PER_HA.get(str(c), 15_000.0) for c in crops], dtype=float
        )

        # Irrigated land outperforms rainfed; irrigation also raises input cost.
        irrigation_yield = np.where(is_irrigated, 1.16, 0.86)
        irrigation_cost = np.where(is_irrigated, 1.13, 0.87)

        # Improved cultivars beat local ones.
        variety_gain = np.where(
            np.char.startswith(varieties.astype(str), "Bt"), 1.28, 1.0
        )
        variety_gain = np.where(
            np.char.find(varieties.astype(str), "Hybrid") >= 0, 1.22, variety_gain
        )

        efficiency = (
            base_yield
            * state_yield_mult
            * np.exp(zone_yield)
            * trend
            * irrigation_yield
            * variety_gain
            * np.exp(rng.normal(0.0, NOISE_SIGMA_YIELD, size=batch))
        )
        efficiency = np.clip(efficiency, 0.05, None)

        # Cost scales sub-linearly with area (economies of scale on fixed costs
        # such as machinery access and land preparation).
        cost_per_ha = (
            base_cost
            * state_cost_mult
            * np.exp(zone_cost)
            * inflation
            * irrigation_cost
            * np.exp(rng.normal(0.0, NOISE_SIGMA_COST, size=batch))
        )
        total_cost = cost_per_ha * np.power(area, 0.92)

        production_quintals = efficiency * area

        # Convert to the reported unit.
        as_tons = np.array([str(u) == "Tons" for u in units])
        production_reported = np.where(as_tons, production_quintals / 20.0,
                                       production_quintals)

        for i in range(batch):
            records.append(
                {
                    "Crop": str(crops[i]),
                    "Variety": str(varieties[i]),
                    "state": str(state_arr[i]),
                    "Quantity": float(np.round(area[i], 2)),
                    "production": float(np.round(production_reported[i], 2)),
                    "Season": str(seasons[i]),
                    "Unit": str(units[i]),
                    "Cost": float(np.round(total_cost[i], 2)),
                    "Recommended Zone": _zone_label(
                        rng,
                        state_arr[i],
                        mandals[i],
                        villages[i],
                        bool(has_mandals_arr[i]),
                    ),
                    "Year": int(years_arr[i]),
                }
            )

    frame = pd.DataFrame.from_records(records[:n_rows])

    if dirty:
        frame = _inject_defects(frame, rng)
    return frame


def _inject_defects(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add realistic data-quality defects for the loader to clean."""
    frame = frame.copy()
    n = len(frame)

    def pick(rate: float) -> np.ndarray:
        return rng.random(n) < rate

    # Inconsistent unit spellings.
    tons_mask = frame["Unit"].eq("Tons")
    frame.loc[tons_mask & pick(0.30), "Unit"] = "tons"
    frame.loc[tons_mask & pick(0.15), "Unit"] = "TONS"
    frame.loc[pick(0.02), "Unit"] = "Quintals"
    frame.loc[tons_mask & pick(0.10), "Unit"] = " Tonne "

    # Whitespace and casing in the string columns.
    for col in ("Crop", "Variety", "Season"):
        idx = np.flatnonzero(pick(0.05))
        frame.loc[idx, col] = "  " + frame.loc[idx, col].astype(str) + " "

    idx = np.flatnonzero(pick(0.04))
    frame.loc[idx, "Variety"] = frame.loc[idx, "Variety"].astype(str).str.lower()

    idx = np.flatnonzero(pick(0.04))
    frame.loc[idx, "state"] = "  " + frame.loc[idx, "state"].astype(str)

    # Missing cells.
    frame.loc[pick(0.020), "Cost"] = np.nan
    frame.loc[pick(0.015), "production"] = np.nan
    frame.loc[pick(0.025), "Variety"] = np.nan
    frame.loc[pick(0.012), "Season"] = np.nan
    frame.loc[pick(0.008), "Unit"] = np.nan

    # Invalid areas (zero / negative), as produced by double-entry slips.
    idx = np.flatnonzero(pick(0.006))
    frame.loc[idx, "Quantity"] = 0.0
    idx = np.flatnonzero(pick(0.003))
    frame.loc[idx, "Quantity"] = -abs(
        rng.uniform(0.1, 50.0, size=len(idx))
    )

    # Entry-error outliers in production.
    idx = np.flatnonzero(pick(0.0035))
    frame.loc[idx, "production"] = frame.loc[idx, "production"].astype(float) * (
        15.0 + rng.random(len(idx)) * 45.0
    )
    idx = np.flatnonzero(pick(0.0025))
    frame.loc[idx, "Cost"] = frame.loc[idx, "Cost"].astype(float) * (
        8.0 + rng.random(len(idx)) * 22.0
    )

    return frame


def summarise(frame: pd.DataFrame) -> str:
    """One-paragraph description used by the CLI and the training report."""
    return (
        f"{len(frame):,} synthetic rows | "
        f"{frame['Crop'].nunique()} crops | "
        f"{frame['state'].nunique()} states/UTs | "
        f"{int(frame['Year'].min())}-{int(frame['Year'].max())} | "
        f"{frame['Recommended Zone'].nunique():,} distinct zones"
    )


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    from config.settings import SYNTHETIC_CSV

    sample = generate_synthetic_dataset(2_000)
    print(summarise(sample))
    print(sample.head(10).to_string())
    sample.to_csv(SYNTHETIC_CSV, index=False)
    print(f"\nwritten -> {SYNTHETIC_CSV}")
