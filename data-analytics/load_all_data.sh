#!/usr/bin/env bash
# Generate the CSVs and import everything into MySQL in one shot.
# Usage: ./load_all_data.sh [--user root] [--password ""] [--host 127.0.0.1] [--port 3306]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GENERATOR="$REPO_ROOT/scratches/data-analytics/generate_test_data.py"
IMPORTER="$SCRIPT_DIR/import_to_mysql.sh"

USER="root"
PASSWORD=""
HOST="127.0.0.1"
PORT="3306"

print_usage() {
  cat <<EOF
Usage: $0 [--user USER] [--password PASSWORD] [--host HOST] [--port PORT]
This script regenerates the CSVs (clean + dirty) and imports them into analytics_demo.
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

echo "1) Regenerating datasets"
python3 "$GENERATOR" --output-dir "$REPO_ROOT/data_output"

MYSQL_CMD=(mysql -u"$USER" -h"$HOST" -P"$PORT")
if [[ -n "$PASSWORD" ]]; then
  MYSQL_CMD+=( -p"$PASSWORD" )
fi

echo "2) Dropping existing schema to ensure clean load"
"${MYSQL_CMD[@]}" <<'EOF'
DROP DATABASE IF EXISTS analytics_demo;
CREATE DATABASE analytics_demo;
EOF

echo "3) Loading data into MySQL"
bash "$IMPORTER" --user "$USER" --password "$PASSWORD" --host "$HOST" --port "$PORT"

echo "Done. Tables in analytics_demo refreshed from freshly generated data."
