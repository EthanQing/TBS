#!/bin/bash
set -e

echo "======================================"
echo "  Train Platform Backend starting..."
echo "======================================"

echo "Checking offline license..."
python3 -c "from train_platform.core.license import assert_valid_license; assert_valid_license()"

# Wait for MySQL to be ready when MYSQL_HOST is configured.
if [ -n "$MYSQL_HOST" ]; then
    echo "Waiting for MySQL ($MYSQL_HOST:${MYSQL_PORT:-3306})..."
    max_retries=30
    count=0
    while ! mysqladmin ping -h"$MYSQL_HOST" -P"${MYSQL_PORT:-3306}" -u"${MYSQL_USER:-root}" -p"${MYSQL_PASSWORD:-}" --silent 2>/dev/null; do
        count=$((count + 1))
        if [ $count -ge $max_retries ]; then
            echo "Error: MySQL connection timed out"
            exit 1
        fi
        echo "  MySQL not ready, waiting... ($count/$max_retries)"
        sleep 2
    done
    echo "MySQL is ready"
fi

# Only the main API service runs migrations and seed data.
if [[ "$*" != *"worker"* ]]; then
    echo "Running database migrations (Alembic)..."
    alembic upgrade head

    echo "Seeding reference data..."
    python3 -c "from train_platform.db.init_db import init_db; init_db()"
else
    echo "Worker mode: skipping migrations (handled by backend)"
    # Give backend a moment to complete migrations.
    sleep 5
fi

echo "======================================"
echo "  Starting service..."
echo "======================================"

exec "$@"
