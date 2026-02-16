#!/usr/bin/env bash
# Import the generated CSV data into MySQL (analytics_demo schema).
# Usage: ./import_to_mysql.sh --user root --password '' --host 127.0.0.1 --port 3306

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_DIR="$REPO_ROOT/data_output"

USER="root"
PASSWORD=""
HOST="127.0.0.1"
PORT="3306"

print_usage() {
  cat <<EOF
Usage: $0 [--user USER] [--password PASSWORD] [--host HOST] [--port PORT]
Default user/password/host/port assume a Homebrew installation (root with empty password).
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --user) USER=$2; shift 2;;
    --password) PASSWORD=$2; shift 2;;
    --host) HOST=$2; shift 2;;
    --port) PORT=$2; shift 2;;
    -h|--help) print_usage;;
    *) echo "Unknown option: $1" >&2; print_usage;;
  esac
done

MYSQL_ARGS=("--local-infile=1" "-u$USER" "-h" "$HOST" "-P" "$PORT")
if [[ -n "$PASSWORD" ]]; then
  MYSQL_ARGS+=("-p$PASSWORD")
fi
MYSQL=(mysql "${MYSQL_ARGS[@]}")

run_sql() {
  echo "Running SQL: $1"
  "${MYSQL[@]}" -e "$1"
}

SQL_SCHEMA=$(cat <<'EOF'
CREATE DATABASE IF NOT EXISTS analytics_demo;
USE analytics_demo;
CREATE TABLE IF NOT EXISTS customers (
  customer_id INT PRIMARY KEY,
  first_name VARCHAR(64),
  last_name VARCHAR(64),
  email VARCHAR(128),
  segment VARCHAR(32),
  country CHAR(2),
  signup_date DATE
);
CREATE TABLE IF NOT EXISTS products (
  product_id INT PRIMARY KEY,
  name VARCHAR(128),
  category VARCHAR(64),
  list_price DECIMAL(10,2),
  launch_date DATE
);
CREATE TABLE IF NOT EXISTS orders (
  order_id INT PRIMARY KEY,
  customer_id INT,
  order_date DATE,
  status VARCHAR(32),
  channel VARCHAR(32),
  payment_method VARCHAR(32),
  order_total DECIMAL(13,2),
  FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
CREATE TABLE IF NOT EXISTS order_items (
  order_id INT,
  product_id INT,
  quantity INT,
  unit_price DECIMAL(10,2),
  amount DECIMAL(13,2),
  FOREIGN KEY (order_id) REFERENCES orders(order_id),
  FOREIGN KEY (product_id) REFERENCES products(product_id)
);
CREATE TABLE IF NOT EXISTS sessions (
  session_id INT PRIMARY KEY,
  customer_id INT,
  start_time DATETIME,
  end_time DATETIME,
  device VARCHAR(32),
  pages INT,
  intention VARCHAR(32),
  FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
EOF
)

run_sql "$SQL_SCHEMA"

import_file() {
  local table=$1
  local file=$2
  echo "Loading $file into $table"
  "${MYSQL[@]}" analytics_demo <<EOF
SET GLOBAL local_infile = 1;
LOAD DATA LOCAL INFILE '$file'
INTO TABLE $table
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;
EOF
}

import_file customers "$DATA_DIR/customers.csv"
import_file products "$DATA_DIR/products.csv"
import_file orders "$DATA_DIR/orders.csv"
import_file order_items "$DATA_DIR/order_items.csv"
import_file sessions "$DATA_DIR/sessions.csv"

echo "Import complete."
