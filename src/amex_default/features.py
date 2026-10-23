"""Disk-partitioned statement aggregation with chronological endpoint values."""

import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import fingerprint, read_json, resolve, save_json

CATEGORICAL = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120",
               "D_126", "D_63", "D_64", "D_66", "D_68"]


def qi(value):
    return '"' + str(value).replace('"', '""') + '"'


def qs(value):
    return "'" + str(value).replace("'", "''") + "'"


def source_sql(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Place the statement file at {path}")
    if path.suffix.lower() == ".parquet":
        return f"read_parquet({qs(path)})"
    if path.suffix.lower() in {".csv", ".gz"}:
        return f"read_csv({qs(path)}, header=true, all_varchar=true)"
    raise ValueError("Use a .csv, .csv.gz or .parquet statement file")


def discover_schema(columns, cfg):
    required = {"customer_ID", "S_2"}
    if not required <= set(columns):
        raise ValueError(f"Statement files need {sorted(required)}")
    predictors = sorted(c for c in columns if re.fullmatch(r"[BDPRS]_\d+", c) and c != "S_2")
    if not predictors:
        raise ValueError("No statement predictors found")
    categorical = [c for c in predictors if c in CATEGORICAL]
    return {"numeric": [c for c in predictors if c not in CATEGORICAL],
            "categorical": categorical, "recipe": cfg["features"]["recipe"],
            "data": cfg["data"]}


def normalized_select(schema):
    expressions = ['CAST("customer_ID" AS VARCHAR) AS "customer_ID"',
                   'CAST("S_2" AS DATE) AS "S_2"']
    for col in schema["numeric"]:
        value = f"CAST({qi(col)} AS DOUBLE)"
        sentinel = schema["data"]["numeric_missing_sentinel"]
        if sentinel is not None:
            value = f"NULLIF({value}, {float(sentinel)})"
        # CSV values and NaN-bearing parquet inputs share one missing-value rule.
        expressions.append(f"CASE WHEN isfinite({value}) THEN {value} END AS {qi(col)}")
    for col in schema["categorical"]:
        value = f"CAST({qi(col)} AS VARCHAR)"
        sentinel = schema["data"]["categorical_missing_sentinel"]
        if sentinel is not None:
            value = f"NULLIF({value}, {qs(sentinel)})"
        expressions.append(f"{value} AS {qi(col)}")
    return ",\n".join(expressions)


def aggregate_sql(schema, source):
    expressions = ["customer_ID", 'CAST(count(*) AS INTEGER) AS "n_statements"',
                   'CAST(date_diff(\'day\', min(S_2), max(S_2)) AS INTEGER) AS "history_days"']
    for col in schema["numeric"]:
        c = qi(col)
        funcs = {"mean": f"avg({c})", "min": f"min({c})", "max": f"max({c})",
                 "last": f"arg_max_null({c}, S_2)"}
        if schema["recipe"] == "extended":
            funcs.update({"std": f"stddev_samp({c})", "first": f"arg_min_null({c}, S_2)",
                          "last_minus_first": f"arg_max_null({c}, S_2)-arg_min_null({c}, S_2)",
                          "last_minus_mean": f"arg_max_null({c}, S_2)-avg({c})",
                          "missing_fraction": f"1.0-count({c})::DOUBLE/count(*)"})
        expressions.extend(f"CAST({value} AS FLOAT) AS {qi(col+'__'+name)}"
                           for name, value in funcs.items())
    for col in schema["categorical"]:
        expressions += [f"arg_max_null({qi(col)}, S_2) AS {qi(col+'__last')}",
                        f"CAST(count(DISTINCT {qi(col)}) AS INTEGER) AS {qi(col+'__nunique')}"]
    return "SELECT " + ",\n".join(expressions) + f" FROM {source} GROUP BY customer_ID ORDER BY customer_ID"


def prepare(cfg, split, overwrite=False):
    """Scan once into customer buckets, then aggregate one bucket at a time."""
    import duckdb

    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    root = resolve(cfg, "features")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / split
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} exists; use --overwrite to rebuild features")
    work = root / (".work-" + uuid.uuid4().hex)
    stage, finished = work / "statements", work / "features"
    finished.mkdir(parents=True)
    source = source_sql(resolve(cfg, split))
    connection = duckdb.connect()
    try:
        connection.execute(f"SET memory_limit = {qs(cfg['processing']['memory_limit'])}")
        connection.execute(f"SET threads = {int(cfg['processing']['threads'])}")
        connection.execute(f"SET temp_directory = {qs(work / 'spill')}")
        connection.execute("SET preserve_insertion_order = false")
        columns = [row[0] for row in connection.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()]
        schema = discover_schema(columns, cfg)
        signature = fingerprint(schema)
        if split == "test":
            train_meta = read_json(root / "train" / "_metadata.json")
            if signature != train_meta["schema_signature"]:
                raise ValueError("Train and test need the same predictor schema and feature recipe")
        connection.execute(f"CREATE VIEW normalized AS SELECT {normalized_select(schema)} FROM {source}")
        buckets = int(cfg["processing"]["partitions"])
        print(f"Partitioning {split} statements into {buckets} customer buckets", flush=True)
        connection.execute(f"COPY (SELECT *, hash(customer_ID) % {buckets} AS bucket FROM normalized) "
                           f"TO {qs(stage)} (FORMAT PARQUET, PARTITION_BY(bucket), COMPRESSION ZSTD)")
        stats = {"statement_rows": 0, "customers": 0, "first_statement": None, "last_statement": None}
        parts = sorted(stage.glob("bucket=*"))
        for i, part in enumerate(parts):
            table = f"read_parquet({qs(part / '*.parquet')}, hive_partitioning=false)"
            nrows, ncustomers, bad, first, last, unique_rows = connection.execute(
                f"SELECT count(*), count(DISTINCT customer_ID), "
                f"count(*) FILTER (WHERE customer_ID IS NULL OR customer_ID='' OR S_2 IS NULL), "
                f"min(S_2), max(S_2), count(DISTINCT (customer_ID, S_2)) FROM {table}"
            ).fetchone()
            if bad or unique_rows != nrows:
                raise ValueError("Each statement needs an ID, a valid date, and a unique customer/date pair")
            stats["statement_rows"] += nrows
            stats["customers"] += ncustomers
            for name, date, func in [("first_statement", first, min), ("last_statement", last, max)]:
                if date is not None:
                    old = stats[name]
                    stats[name] = str(date) if old is None else func(old, str(date))
            query = aggregate_sql(schema, table)
            connection.execute(f"COPY ({query}) TO {qs(finished / f'part-{i:03d}.parquet')} "
                               "(FORMAT PARQUET, COMPRESSION ZSTD)")
            print(f"Aggregated bucket {i+1}/{len(parts)}", flush=True)
        if not stats["customers"]:
            raise ValueError("Statement file is empty")
        raw_path = resolve(cfg, split)
        metadata = {"split": split, "prepare_id": uuid.uuid4().hex, "schema": schema,
                    "schema_signature": signature, "statistics": stats,
                    "source": {"path": str(raw_path), "size_bytes": raw_path.stat().st_size,
                               "mtime_ns": raw_path.stat().st_mtime_ns},
                    "created_at_utc": datetime.now(timezone.utc).isoformat()}
        save_json(finished / "_metadata.json", metadata)
        if destination.exists():
            shutil.rmtree(destination)
        finished.rename(destination)
        save_json(root / f"{split}_metadata.json", metadata)
        return metadata
    finally:
        connection.close()
        shutil.rmtree(work, ignore_errors=True)
