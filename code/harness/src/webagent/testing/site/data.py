"""Deterministic local-site data generation.

The generator is deliberately pure: the same ``seed`` and parameters always
produce the same products, posts, hidden roles and display order. It never
reads the clock, the environment or a remote service.

Public records (products, posts) are what the website renders. The hidden
truth (roles, relevance/tone labels) is stored in a separate SQLite table and
is only read by the evaluator, never by a page handler.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Mapping

_ALL_CATEGORIES = (
    "Audio",
    "Books",
    "Camping",
    "Fitness",
    "Garden",
    "Home Office",
    "Kitchen",
    "Travel",
)

_ADJECTIVES = (
    "Alder",
    "Amber",
    "Brisk",
    "Cedar",
    "Cobalt",
    "Copper",
    "Drift",
    "Ember",
    "Fern",
    "Fjord",
    "Glacier",
    "Harbor",
    "Hazel",
    "Indigo",
    "Juniper",
    "Lumen",
    "Maple",
    "Meadow",
    "Moss",
    "Nimbus",
    "Oak",
    "Orchard",
    "Pine",
    "Quartz",
    "Ridge",
    "River",
    "Sable",
    "Saffron",
    "Slate",
    "Solstice",
    "Tidal",
    "Timber",
    "Umber",
    "Vale",
    "Willow",
    "Zephyr",
)

_MATERIALS = (
    "Canvas",
    "Ceramic",
    "Cotton",
    "Glass",
    "Leather",
    "Linen",
    "Mesh",
    "Oak",
    "Recycled",
    "Steel",
    "Stoneware",
    "Titanium",
    "Walnut",
    "Wool",
)

_NOUNS = (
    "Backpack",
    "Bottle",
    "Chair",
    "Clock",
    "Desk Lamp",
    "Field Journal",
    "Filter",
    "Kettle",
    "Lantern",
    "Mug",
    "Notebook",
    "Pack",
    "Pan",
    "Pouch",
    "Rug",
    "Shelf",
    "Speaker",
    "Stool",
    "Tent",
    "Thermos",
    "Tote",
    "Tray",
    "Vase",
    "Wallet",
)

_SUMMARIES = (
    "Built for everyday use with a compact, repairable design.",
    "A practical pick for small spaces and weekend trips.",
    "Tested for balance, durability and easy cleaning.",
    "Simple materials, clear care instructions and a two-year warranty.",
    "Designed to pack flat and travel without rattling.",
    "A quiet, low-maintenance option for a shared workspace.",
)

_POST_TOPICS = (
    "Trail notes from a wet weekend",
    "Small kitchen, better mornings",
    "A desk setup that finally fits",
    "What survived the coastal wind",
    "Three useful field repairs",
    "Packing list for a two-night trip",
    "A quiet corner for reading",
    "Comparing two compact lanterns",
    "Garden shelf after one season",
    "A thermos that keeps tea hot",
    "Storage ideas for a narrow hall",
    "Notes on repairing a canvas tote",
    "Morning routine with less clutter",
    "A lightweight chair for the balcony",
    "What I changed after a month",
    "Keeping tools dry on the trail",
    "A simple shelf for the studio",
    "Two weeks with the travel mug",
    "A small fix for a wobbly stool",
    "Budget setup for a guest room",
)

_POST_AUTHORS = (
    "Ari",
    "Bea",
    "Chen",
    "Dana",
    "Eli",
    "Faye",
    "Gus",
    "Hana",
    "Ivo",
    "June",
    "Kai",
    "Lena",
    "Milo",
    "Nora",
    "Owen",
    "Pia",
    "Quinn",
    "Rosa",
    "Sami",
    "Tess",
)

_POST_TONES = ("positive", "neutral", "negative")
_POST_RELEVANCE = ("relevant", "somewhat_relevant", "not_relevant")

_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


@dataclass(frozen=True)
class Product:
    id: str
    sku: str
    name: str
    category: str
    price_cents: int
    rating: float
    stock: int
    summary: str
    description: str
    position: int

    @property
    def price_label(self) -> str:
        return f"${self.price_cents / 100:.2f}"

    def matches(self, conditions: Mapping[str, Any]) -> bool:
        """Public predicates used by generator self-checks, not by the website."""

        category = conditions.get("category")
        max_price = conditions.get("max_price_cents")
        min_rating = conditions.get("min_rating")
        if category is not None and self.category != category:
            return False
        if max_price is not None and self.price_cents > int(max_price):
            return False
        if min_rating is not None and self.rating < float(min_rating):
            return False
        return self.stock > 0


@dataclass(frozen=True)
class Post:
    id: str
    title: str
    author: str
    body: str
    likes: int
    comments: int
    tags: tuple[str, ...]
    created_label: str
    position: int
    relevance: str = "unknown"
    tone: str = "unknown"


@dataclass(frozen=True)
class HiddenRecord:
    item_type: str
    item_id: str
    role: str
    relevance: str | None = None
    tone: str | None = None
    expected_group: str | None = None


@dataclass(frozen=True)
class Dataset:
    seed: int
    products: tuple[Product, ...]
    posts: tuple[Post, ...]
    hidden: tuple[HiddenRecord, ...] = field(default_factory=tuple)
    generator_params: Mapping[str, Any] = field(default_factory=dict)


def _code(rng: random.Random, used: set[str], length: int = 5) -> str:
    while True:
        candidate = "".join(rng.choice(_CODE_ALPHABET) for _ in range(length))
        if candidate not in used:
            used.add(candidate)
            return candidate


def _price(rng: random.Random) -> int:
    return rng.randrange(1200, 12000) // 100 * 100 + 99


def _matching_product_values(
    rng: random.Random, conditions: Mapping[str, Any]
) -> dict[str, Any]:
    max_price = int(conditions["max_price_cents"])
    low = max(500, max_price - 3600)
    high = max(low + 100, max_price - 100)
    price = rng.randrange(low, high) // 100 * 100 + 99
    min_rating = float(conditions["min_rating"])
    rating = round(rng.uniform(min_rating, 5.0), 1)
    stock = rng.randrange(1, 40)
    return {"price_cents": price, "rating": rating, "stock": stock}


def _decoy_product_values(
    rng: random.Random, conditions: Mapping[str, Any]
) -> dict[str, Any]:
    max_price = int(conditions["max_price_cents"])
    min_rating = float(conditions["min_rating"])
    values = _matching_product_values(rng, conditions)
    mode = rng.choice(("price", "rating", "stock"))
    if mode == "price":
        values["price_cents"] = max_price + rng.randrange(500, 4000)
    elif mode == "rating":
        values["rating"] = round(max(3.0, min_rating - rng.uniform(0.2, 0.9)), 1)
    else:
        values["stock"] = 0
    return values


def _positions(
    rng: random.Random,
    total: int,
    count: int,
    page_size: int,
    reserved: set[int] | None = None,
) -> list[int]:
    """Pick positions spread over the first pages, avoiding reserved slots."""

    if count <= 0 or total <= 0:
        return []
    reserved = set(reserved or ())
    positions: list[int] = []
    pages = max(1, math.ceil(total / max(1, page_size)))
    for page in range(min(pages, count)):
        candidates = [
            index
            for index in range(page * page_size, min(total, (page + 1) * page_size))
            if index not in reserved and index not in positions
        ]
        if candidates:
            positions.append(rng.choice(candidates))
    while len(positions) < min(count, total - len(reserved)):
        candidates = [
            index
            for index in range(total)
            if index not in reserved and index not in positions
        ]
        if not candidates:
            break
        positions.append(rng.choice(candidates))
    return positions


def _product_description(
    name: str, category: str, summary: str, rng: random.Random
) -> str:
    care = rng.choice(
        (
            "Wipe clean with a damp cloth.",
            "Hand wash and air dry.",
            "Recyclable packaging; no tools needed.",
            "Ships flat with a printed care card.",
        )
    )
    return (
        f"{name} is part of the {category.lower()} collection. {summary} "
        f"Each unit is checked before packing. {care}"
    )


def _generate_products(
    rng: random.Random,
    params: Mapping[str, Any],
    hidden: list[HiddenRecord],
) -> tuple[Product, ...]:
    count = int(params.get("products", 32))
    page_size = int(params.get("page_size", 8))
    l01 = params.get("l01_conditions")
    l02 = params.get("l02_conditions")
    l06 = params.get("l06_conditions")
    l07 = params.get("l07_conditions")
    special_conditions = (l01, l02, l06, l07)
    category_pool = [
        category
        for category in _ALL_CATEGORIES
        if category
        not in {str((conditions or {}).get("category")) for conditions in special_conditions}
    ]
    used_codes: set[str] = set()
    records: list[dict[str, Any]] = []
    for index in range(count):
        code = _code(rng, used_codes)
        adjective = rng.choice(_ADJECTIVES)
        material = rng.choice(_MATERIALS)
        noun = rng.choice(_NOUNS)
        name = f"{adjective} {material} {noun} {code}"
        category = rng.choice(category_pool)
        summary = rng.choice(_SUMMARIES)
        records.append(
            {
                "id": f"P-{code}",
                "sku": f"SKU-{code}",
                "name": name,
                "category": category,
                "price_cents": _price(rng),
                "rating": round(rng.uniform(3.4, 5.0), 1),
                "stock": rng.randrange(0, 60),
                "summary": summary,
                "description": "",
                "position": index,
            }
        )

    if l02:
        matches = int(params.get("l02_count", 5))
        positions = _positions(rng, count, matches, page_size)
        for position in positions:
            records[position].update(_matching_product_values(rng, l02))
            records[position]["category"] = str(l02["category"])
            hidden.append(
                HiddenRecord(
                    "product",
                    records[position]["id"],
                    "l02_match",
                    expected_group="l02",
                )
            )
        decoys = int(params.get("l02_decoys", 3))
        decoy_positions = _positions(
            rng, count, decoys, page_size, reserved=set(positions)
        )
        for position in decoy_positions:
            records[position].update(_decoy_product_values(rng, l02))
            records[position]["category"] = str(l02["category"])
        reserved = set(positions) | set(decoy_positions)
    else:
        reserved = set()

    if l01:
        candidates = [index for index in range(count) if index not in reserved]
        position = rng.choice(candidates)
        records[position].update(_matching_product_values(rng, l01))
        records[position]["category"] = str(l01["category"])
        hidden.append(
            HiddenRecord(
                "product",
                records[position]["id"],
                "l01_target",
                expected_group="l01",
            )
        )
        decoys = int(params.get("l01_decoys", 2))
        decoy_positions = _positions(
            rng, count, decoys, page_size, reserved=reserved | {position}
        )
        for decoy_position in decoy_positions:
            records[decoy_position].update(_decoy_product_values(rng, l01))
            records[decoy_position]["category"] = str(l01["category"])
        reserved = reserved | {position} | set(decoy_positions)

    # Control cases (L06/L07) need their own unique public target and role so
    # the external judge is not borrowing another case's hidden truth.
    for prefix, conditions, role in (
        ("l06", l06, "l06_target"),
        ("l07", l07, "l07_target"),
    ):
        if not conditions:
            continue
        candidates = [index for index in range(count) if index not in reserved]
        position = rng.choice(candidates)
        records[position].update(_matching_product_values(rng, conditions))
        records[position]["category"] = str(conditions["category"])
        hidden.append(
            HiddenRecord(
                "product",
                records[position]["id"],
                role,
                expected_group=prefix,
            )
        )
        decoys = int(params.get(f"{prefix}_decoys", 2))
        decoy_positions = _positions(
            rng, count, decoys, page_size, reserved=reserved | {position}
        )
        for decoy_position in decoy_positions:
            records[decoy_position].update(_decoy_product_values(rng, conditions))
            records[decoy_position]["category"] = str(conditions["category"])
        reserved = reserved | {position} | set(decoy_positions)

    products: list[Product] = []
    for record in records:
        name = str(record["name"])
        category = str(record["category"])
        summary = str(record["summary"])
        description = _product_description(name, category, summary, rng)
        products.append(
            Product(
                id=str(record["id"]),
                sku=str(record["sku"]),
                name=name,
                category=category,
                price_cents=int(record["price_cents"]),
                rating=float(record["rating"]),
                stock=int(record["stock"]),
                summary=summary,
                description=description,
                position=int(record["position"]),
            )
        )
    return tuple(products)


def _generate_posts(
    rng: random.Random,
    params: Mapping[str, Any],
    hidden: list[HiddenRecord],
) -> tuple[Post, ...]:
    count = int(params.get("posts", 20))
    l03 = params.get("l03")
    role = str((l03 or {}).get("role", "l03_target")) if l03 else None
    used_codes: set[str] = set()
    posts: list[Post] = []
    for index in range(count):
        code = _code(rng, used_codes, length=4)
        topic = rng.choice(_POST_TOPICS)
        adjective = rng.choice(_ADJECTIVES)
        noun = rng.choice(_NOUNS)
        title = f"{topic} — {adjective} {noun} ({code})"
        author = rng.choice(_POST_AUTHORS)
        relevance = rng.choice(_POST_RELEVANCE)
        tone = rng.choice(_POST_TONES)
        sentences = [
            f"I picked up the {adjective.lower()} {noun.lower()} two weeks ago and "
            f"used it around {rng.choice(('the kitchen', 'a campsite', 'the balcony', 'a small studio'))}.",
            rng.choice(
                (
                    "The finish has held up well after repeated use.",
                    "It is lighter than expected, which helped on the walk back.",
                    "Setup took under ten minutes with no extra tools.",
                    "The size is right for a narrow shelf, though the lid is tight.",
                    "I would buy it again for a guest room or a short trip.",
                    "One seam needed a quick stitch after the first week.",
                )
            ),
            rng.choice(
                (
                    "Happy to answer questions if anyone is comparing options.",
                    "I have not tested it in heavy rain yet.",
                    "Photos are from the second weekend, not the first unboxing.",
                    "This is not a sponsored post; I bought it at the local shop.",
                )
            ),
        ]
        body = " ".join(sentences)
        tags = tuple(
            sorted(
                {
                    rng.choice(("gear", "home", "field-notes", "repair", "budget", "travel")),
                    rng.choice(("review", "question", "tip", "photo")),
                }
            )
        )
        day = 1 + (index * 7 + rng.randrange(0, 7)) % 28
        created_label = f"2026-10-{day:02d}"
        posts.append(
            Post(
                id=f"T-{code}",
                title=title,
                author=author,
                body=body,
                likes=rng.randrange(0, 180),
                comments=rng.randrange(0, 24),
                tags=tags,
                created_label=created_label,
                position=index,
                relevance=relevance,
                tone=tone,
            )
        )
        if role:
            hidden.append(
                HiddenRecord(
                    "post",
                    posts[-1].id,
                    role,
                    relevance=relevance,
                    tone=tone,
                    expected_group="l03",
                )
            )
    return tuple(posts)


def generate_dataset(
    seed: int, params: Mapping[str, Any] | None = None
) -> Dataset:
    """Build one deterministic case dataset from a seed."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    params = dict(params or {})
    product_rng = random.Random(f"webagent:{seed}:products")
    post_rng = random.Random(f"webagent:{seed}:posts")
    hidden: list[HiddenRecord] = []
    products = _generate_products(product_rng, params, hidden)
    posts = _generate_posts(post_rng, params, hidden)

    l01 = params.get("l01_conditions")
    l02 = params.get("l02_conditions")
    if l01:
        matches = [product for product in products if product.matches(l01)]
        if len(matches) != 1:
            raise ValueError(
                f"data generator expected exactly one L01 match, got {len(matches)}"
            )
    if l02:
        matches = [product for product in products if product.matches(l02)]
        expected = int(params.get("l02_count", 5))
        if len(matches) != expected:
            raise ValueError(
                f"data generator expected {expected} L02 matches, got {len(matches)}"
            )
    return Dataset(
        seed=seed,
        products=products,
        posts=posts,
        hidden=tuple(hidden),
        generator_params=params,
    )


__all__ = [
    "Dataset",
    "HiddenRecord",
    "Post",
    "Product",
    "generate_dataset",
]
