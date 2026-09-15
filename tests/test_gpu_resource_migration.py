import importlib.util
from pathlib import Path

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect


def test_gpu_migration_upgrade_and_downgrade(tmp_path):
    versions = Path(__file__).parents[1] / "train_platform/db/migrations/versions"
    migration_path, = versions.glob("0024*.py")
    spec = importlib.util.spec_from_file_location("gpu_migration", migration_path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    engine = create_engine(f"sqlite:///{tmp_path / 'migration.db'}")
    with engine.begin() as connection:
        # The new revision references the existing run table; no historical run
        # should acquire a request during the upgrade.
        connection.exec_driver_sql("CREATE TABLE training_runs (run_id VARCHAR(36) PRIMARY KEY)")
        connection.exec_driver_sql("INSERT INTO training_runs VALUES ('legacy-run')")
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
            tables = set(inspect(connection).get_table_names())
            assert {"gpu_devices", "gpu_worker_instances", "gpu_worker_observations",
                    "training_run_resource_requests"} <= tables
            assert connection.exec_driver_sql("SELECT COUNT(*) FROM training_run_resource_requests").scalar() == 0
            constraints = inspect(connection).get_unique_constraints("gpu_worker_observations")
            assert any(set(item["column_names"]) == {"instance_id", "gpu_uuid"} for item in constraints)
            migration.downgrade()
        assert inspect(connection).get_table_names() == ["training_runs"]
        assert connection.exec_driver_sql("SELECT run_id FROM training_runs").scalar() == "legacy-run"
    engine.dispose()


def test_managed_gpu_migration_preserves_legacy_runs(tmp_path):
    versions = Path(__file__).parents[1] / "train_platform/db/migrations/versions"
    revisions = []
    for prefix in ("0024", "0025"):
        migration_path, = versions.glob(f"{prefix}*.py")
        spec = importlib.util.spec_from_file_location(prefix, migration_path)
        migration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(migration)
        revisions.append(migration)
    engine = create_engine(f"sqlite:///{tmp_path / 'managed-migration.db'}")
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE training_runs (run_id VARCHAR(36) PRIMARY KEY)")
        connection.exec_driver_sql("INSERT INTO training_runs VALUES ('legacy-run')")
        with Operations.context(MigrationContext.configure(connection)):
            for migration in revisions:
                migration.upgrade()
            assert connection.exec_driver_sql("SELECT COUNT(*) FROM gpu_allocations").scalar() == 0
            assert connection.exec_driver_sql("SELECT current_allocation_id FROM training_runs").scalar() is None
            constraints = inspect(connection).get_unique_constraints("gpu_allocations")
            assert any(item["column_names"] == ["active_run_id"] for item in constraints)
            foreign_keys = inspect(connection).get_foreign_keys("gpu_allocations")
            assert all(item["options"].get("ondelete") == "RESTRICT" for item in foreign_keys)
            for migration in reversed(revisions):
                migration.downgrade()
        assert connection.exec_driver_sql("SELECT run_id FROM training_runs").scalar() == "legacy-run"
    engine.dispose()
