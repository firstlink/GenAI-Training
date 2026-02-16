"""Generate CSV files for realistic analytics scenarios."""

from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data_output"


FIRST_NAMES = [
    "Alex",
    "Bailey",
    "Charlie",
    "Dana",
    "Emery",
    "Finn",
    "Glen",
    "Harper",
    "Indigo",
    "Jules",
]

LAST_NAMES = [
    "Adams",
    "Bennett",
    "Chao",
    "Diaz",
    "Ellis",
    "Farrell",
    "Grant",
    "Hughes",
    "Ito",
    "Jenkins",
]

SEGMENTS = ["Startup", "SMB", "Enterprise", "Education", "Non-profit"]
ORDER_STATUS = ["completed", "returned", "abandoned", "in_progress"]
CHANNELS = ["web", "mobile", "partner", "call_center"]
DEVICE_TYPES = ["desktop", "mobile", "tablet", "tv"]
PAYMENT_METHODS = ["card", "ACH", "paypal", "gift_card"]
PRODUCT_CATEGORIES = [
    "AI Tools",
    "Cloud Storage",
    "Security",
    "Analytics",
    "Productivity",
]


def date_range(start: datetime, end: datetime) -> Iterator[datetime]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def round_two(value: float) -> float:
    return round(value, 2)


def generate_customers(num_customers: int) -> list[dict]:
    customers = []
    for cust_id in range(1, num_customers + 1):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        customers.append(
            {
                "customer_id": cust_id,
                "first_name": first,
                "last_name": last,
                "email": f"{first.lower()}.{last.lower()}{cust_id}@example.com",
                "segment": random.choice(SEGMENTS),
                "country": random.choice(["US", "CA", "UK", "DE", "AU", "IN", "BR"]),
                "signup_date": (datetime.now() - timedelta(days=random.randint(30, 720))).date().isoformat(),
            }
        )
    return customers


def generate_products(num_products: int) -> list[dict]:
    products = []
    for prod_id in range(1, num_products + 1):
        category = random.choice(PRODUCT_CATEGORIES)
        products.append(
            {
                "product_id": prod_id,
                "name": f"{category} Suite {prod_id}",
                "category": category,
                "list_price": round_two(random.uniform(29, 399)),
                "launch_date": (datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1500))).date().isoformat(),
            }
        )
    return products


def pick_customer(customers: list[dict]) -> dict:
    return random.choice(customers)


def pick_products(products: list[dict], max_items: int = 4) -> list[dict]:
    count = random.randint(1, max_items)
    return random.sample(products, count)


def generate_orders(
    customers: list[dict],
    products: list[dict],
    start_date: datetime,
    end_date: datetime,
    num_orders: int,
) -> tuple[list[dict], list[dict]]:
    orders = []
    items = []
    order_id = 1
    product_ids = [p["product_id"] for p in products]

    all_dates = list(date_range(start_date, end_date))
    for _ in range(num_orders):
        customer = pick_customer(customers)
        order_date = random.choice(all_dates)
        product_selection = pick_products(products)
        order_items = []
        total = 0.0
        for product in product_selection:
            quantity = random.randint(1, 5)
            price = product["list_price"] * random.uniform(0.8, 1.1)
            amount = round_two(price * quantity)
            total += amount
            order_items.append(
                {
                    "order_id": order_id,
                    "product_id": product["product_id"],
                    "quantity": quantity,
                    "unit_price": round_two(price),
                    "amount": amount,
                }
            )
        orders.append(
            {
                "order_id": order_id,
                "customer_id": customer["customer_id"],
                "order_date": order_date.isoformat(),
                "status": random.choices(ORDER_STATUS, weights=[70, 10, 15, 5])[0],
                "channel": random.choice(CHANNELS),
                "payment_method": random.choice(PAYMENT_METHODS),
                "order_total": round_two(total),
            }
        )
        items.extend(order_items)
        order_id += 1

    return orders, items


def generate_sessions(customers: list[dict], num_sessions: int) -> list[dict]:
    sessions = []
    for session_id in range(1, num_sessions + 1):
        customer = pick_customer(customers)
        start = datetime.now() - timedelta(days=random.randint(0, 365))
        duration = timedelta(minutes=random.randint(1, 45))
        sessions.append(
            {
                "session_id": session_id,
                "customer_id": customer["customer_id"],
                "start_time": start.isoformat(),
                "end_time": (start + duration).isoformat(),
                "device": random.choice(DEVICE_TYPES),
                "pages": random.randint(1, 25),
                "intention": random.choice(["browse", "purchase", "support", "research"]),
            }
        )
    return sessions


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def corrupt_email(email: str) -> str:
    if random.random() < 0.5:
        return email.replace("@", "_at_")
    return email


def inject_customer_anomalies(customers: list[dict], base_customers: list[dict]) -> None:
    if not base_customers:
        return
    anchor = base_customers[0]
    next_id = base_customers[-1]["customer_id"] + 1
    customers.append(
        {
            **anchor,
            "customer_id": next_id,
            "segment": "Unknown",
            "email": corrupt_email(anchor["email"]),
            "country": "US",
        }
    )
    customers.append(
        {
            "customer_id": next_id + 1,
            "first_name": "",
            "last_name": base_customers[-1]["last_name"],
            "email": "no-at-example.com",
            "segment": "",
            "country": "",
            "signup_date": "",
        }
    )
    customers.append(
        {
            **base_customers[1],
            "customer_id": next_id + 2,
            "email": base_customers[1]["email"].upper(),
            "first_name": base_customers[1]["first_name"].lower(),
            "segment": "  smb  ",
        }
    )


def inject_order_anomalies(
    orders: list[dict],
    items: list[dict],
    base_orders: list[dict],
    products: list[dict],
) -> None:
    if not base_orders or len(products) < 3:
        return
    sample = random.choice(base_orders)
    next_order_id = base_orders[-1]["order_id"] + 1

    dup = {**sample, "order_id": next_order_id, "channel": "web"}
    orders.append(dup)
    items.append(
        {
            "order_id": dup["order_id"],
            "product_id": products[0]["product_id"],
            "quantity": 1,
            "unit_price": products[0]["list_price"],
            "amount": products[0]["list_price"],
        }
    )

    future = {
        **sample,
        "order_id": next_order_id + 1,
        "order_date": (datetime.now() + timedelta(days=30)).date().isoformat(),
        "order_total": -round_two(sample["order_total"]),
    }
    orders.append(future)
    items.append(
        {
            "order_id": future["order_id"],
            "product_id": products[1]["product_id"],
            "quantity": 2,
            "unit_price": products[1]["list_price"],
            "amount": round_two(products[1]["list_price"] * 2),
        }
    )

    blank = {**sample, "order_id": next_order_id + 2, "status": "", "channel": "***"}
    orders.append(blank)
    items.append(
        {
            "order_id": blank["order_id"],
            "product_id": products[2]["product_id"],
            "quantity": 1,
            "unit_price": products[2]["list_price"],
            "amount": products[2]["list_price"],
        }
    )


def inject_session_anomalies(sessions: list[dict], base_sessions: list[dict]) -> None:
    if not base_sessions:
        return
    base = random.choice(base_sessions)
    sessions.append({**base, "session_id": base["session_id"], "start_time": base["end_time"], "end_time": base["start_time"]})
    sessions.append({**base, "session_id": base["session_id"] + 1, "pages": -5, "device": "unknown"})
    sessions.append({**base, "session_id": base["session_id"] + 2, "intention": ""})


def main() -> None:
    parser = argparse.ArgumentParser(description="Create CSV test data for data analytics demos.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write CSV files.",
    )
    parser.add_argument("--customers", type=int, default=2000)
    parser.add_argument("--products", type=int, default=150)
    parser.add_argument("--orders", type=int, default=12000)
    parser.add_argument("--sessions", type=int, default=18000)
    args = parser.parse_args()

    random.seed(42)

    customers = generate_customers(args.customers)
    products = generate_products(args.products)
    orders, items = generate_orders(
        customers,
        products,
        start_date=datetime.now() - timedelta(days=365),
        end_date=datetime.now(),
        num_orders=args.orders,
    )
    sessions = generate_sessions(customers, args.sessions)
    base_customers = customers.copy()
    base_orders = orders.copy()
    base_sessions = sessions.copy()
    inject_customer_anomalies(customers, base_customers)
    inject_order_anomalies(orders, items, base_orders, products)
    inject_session_anomalies(sessions, base_sessions)

    output_root = Path(args.output_dir)
    write_csv(output_root / "customers.csv", customers)
    write_csv(output_root / "products.csv", products)
    write_csv(output_root / "orders.csv", orders)
    write_csv(output_root / "order_items.csv", items)
    write_csv(output_root / "sessions.csv", sessions)

    print("Generated datasets (includes injected anomalies).")
    print(f" - customers: {len(customers)}")
    print(f" - products: {len(products)}")
    print(f" - orders: {len(orders)}")
    print(f" - order_items: {len(items)}")
    print(f" - sessions: {len(sessions)}")


if __name__ == "__main__":
    main()
