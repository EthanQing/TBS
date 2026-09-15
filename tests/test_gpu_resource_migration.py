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
