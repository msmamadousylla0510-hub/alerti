"""
Mapping des quartiers de Bamako vers les 6 communes officielles.
Permet de relier les prédictions LSTM (niveau commune) aux quartiers affichés dans le dashboard.
"""

from __future__ import annotations

import unicodedata
from typing import Dict, List, Optional

BAMAKO_COMMUNE_NEIGHBORHOODS: Dict[str, List[str]] = {
    "Commune I": [
        "Banconi",
        "Banconi Sikoro",
        "Banconi-Plateau",
        "Sikoro",
        "Sikoro-Farada",
        "Korofina Nord",
        "Korofina Sud",
        "Korofina-Sud",
        "Fadjiguila",
        "Djelibougou",
        "Djélibougou",
    ],
    "Commune II": [
        "Niaréla",
        "Niarela",
        "Sans Fil",
        "Quinzambougou",
        "Quimzambougou",
        "Bakaribougou",
        "Bougouba",
        "Zone Industrielle",
        "Hippodrome",
        "TSF Campement des pêcheurs",
        "Campement des pêcheurs TSF",
        "Bozola campement des pêcheurs",
        "Medina Coura",
        "Médina Coura",
        "Ngolonina",
    ],
    "Commune III": [
        "Koulouba",
        "Sogonanfing",
        "Sogonafing",
        "Badialan III",
        "Badialan I",
        "Badalan I",
        "Bamako Coura",
        "Bamako-coura",
        "Niomirambougou",
        "Quartier du fleuve",
        "Ouolofobougou Bolibabana",
        "Ntomikorobougou",
        "N'Tomikorobougou",
        "Darsalam",
        "Dar Salam",
        "Centre-ville BDM",
        "Enceinte Base B",
        "Point G",
    ],
    "Commune IV": [
        "Kalabambougou",
        "Lafiabougou Bougoudani",
        "Lafiabougou",
        "Taliko Tietieni",
        "Taliko",
        "Sébénikoro",
        "Sebenikoro",
        "Sebenikororo SEMA 2",
        "Woyowayanko",
    ],
    "Commune V": [
        "Lafiabougou près phar",
        "Lafiabougou Phar",
        "Sebenikoro (Ilot)",
        "Djicoroni Para",
        "Djicoroni para flabougou",
        "Bacodjicoroni",
        "Bacodjicoroni Heremakono",
        "Sabalibougou",
        "Kalaban Coura",
        "Badalabougou",
        "Daoudabougou",
        "Torokorobougou",
    ],
    "Commune VI": [
        "Yirimadio",
        "Missabougou",
        "Magnambougou",
        "Magnambougou Bada",
        "Sirakoro Méguetana",
        "Sirakoro Megetana",
        "Dianeguela",
        "Dianéguela",
    ],
}

# Alias textuels pour reconnaître les communes directement (ex: "commune 1")
COMMUNE_ALIASES: Dict[str, str] = {
    "commune 1": "Commune I",
    "commune i": "Commune I",
    "commune 2": "Commune II",
    "commune ii": "Commune II",
    "commune 3": "Commune III",
    "commune iii": "Commune III",
    "commune 4": "Commune IV",
    "commune iv": "Commune IV",
    "commune 5": "Commune V",
    "commune v": "Commune V",
    "commune 6": "Commune VI",
    "commune vi": "Commune VI",
}


def _normalize_text(value: str) -> str:
    """Lowercase, retire les accents et espaces multiples pour faciliter les correspondances."""
    if not value:
        return ""
    nfkd = unicodedata.normalize("NFKD", value)
    without_accents = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    cleaned = (
        without_accents.replace("-", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace("’", " ")
        .replace("'", " ")
        .replace(",", " ")
        .replace("/", " ")
    )
    cleaned = " ".join(cleaned.lower().split())
    return cleaned


NEIGHBORHOOD_TO_COMMUNE: Dict[str, str] = {}
for commune_name, neighborhoods in BAMAKO_COMMUNE_NEIGHBORHOODS.items():
    for raw_name in neighborhoods:
        normalized = _normalize_text(raw_name)
        if not normalized:
            continue
        NEIGHBORHOOD_TO_COMMUNE[normalized] = commune_name
        # Ajouter variante sans mentions additionnelles (campement, etc.)
        base = normalized.replace("campement des pecheurs", "").replace("campement pecheurs", "")
        base = base.replace("pres", "").replace("proche", "").strip()
        if base and base not in NEIGHBORHOOD_TO_COMMUNE:
            NEIGHBORHOOD_TO_COMMUNE[base] = commune_name


def get_commune_from_neighborhood(neighborhood_name: str) -> Optional[str]:
    """Retourne la commune associée à un quartier (ou None si inconnu)."""
    if not neighborhood_name:
        return None
    normalized = _normalize_text(neighborhood_name)
    if normalized in NEIGHBORHOOD_TO_COMMUNE:
        return NEIGHBORHOOD_TO_COMMUNE[normalized]
    return None


def resolve_neighborhood_from_localite(raw_name: str) -> Optional[str]:
    """
    Convertit une localité capteur (ex. « Quartier de Sebenikoro, Bamako »)
    en nom de quartier reconnu par le registre Bamako.
    """
    if not raw_name:
        return None
    normalized = _normalize_text(raw_name)
    if normalized in NEIGHBORHOOD_TO_COMMUNE:
        return normalized

    stripped = normalized
    for prefix in ("quartier de ", "quartier ", "zone ", "secteur "):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :].strip()
    for suffix in (", bamako", " bamako", " mali"):
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)].strip()

    if stripped in NEIGHBORHOOD_TO_COMMUNE:
        return stripped

    # Correspondance partielle (ex. « sebenikoro yirimadio » → sebenikoro)
    for key in NEIGHBORHOOD_TO_COMMUNE:
        if len(key) >= 5 and (key in stripped or stripped in key):
            return key
    return None


def normalize_commune_name(commune_name: Optional[str]) -> Optional[str]:
    """Normalise les chaînes 'commune 1', 'Commune I', etc. pour coller aux clés officielles."""
    if not commune_name:
        return None
    normalized = _normalize_text(commune_name)
    if normalized in COMMUNE_ALIASES:
        return COMMUNE_ALIASES[normalized]
    # Essayer de matcher la forme exacte (Commune I)
    for official in BAMAKO_COMMUNE_NEIGHBORHOODS.keys():
        if normalized == _normalize_text(official):
            return official
    return None


def list_all_neighborhoods() -> List[str]:
    neighborhoods: List[str] = []
    for names in BAMAKO_COMMUNE_NEIGHBORHOODS.values():
        neighborhoods.extend(names)
    return sorted(set(neighborhoods))


__all__ = [
    "BAMAKO_COMMUNE_NEIGHBORHOODS",
    "NEIGHBORHOOD_TO_COMMUNE",
    "COMMUNE_ALIASES",
    "get_commune_from_neighborhood",
    "resolve_neighborhood_from_localite",
    "normalize_commune_name",
    "list_all_neighborhoods",
]

