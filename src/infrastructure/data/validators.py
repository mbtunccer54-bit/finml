"""Data contracts, compiled from the domain feature contract.

The pandera schemas here are *derived* from :data:`domain.entities.FEATURE_CONTRACT`
rather than written by hand. A field added to the domain contract is validated
automatically, and there is no second list of column names to drift out of sync
with the first.

Validation runs lazily so a bad batch reports every violated constraint at once
instead of only the first — a reviewer chasing a data incident needs the whole
picture, not the first row that failed.
"""

from __future__ import annotations

from typing import Any, Final

import pandas as pd
import pandera.pandas as pa

from domain.entities import (
    DATE_COLUMN,
    ENTITY_ID_COLUMN,
    FEATURE_CONTRACT,
    MACRO_VARIABLES,
    TARGET_COLUMN,
)
from domain.exceptions import SchemaValidationError
from domain.value_objects import FeatureContract, FieldSpec, ScalarDType
from infrastructure.logging import get_logger

__all__ = [
    "INFERENCE_SCHEMA",
    "MACRO_SCHEMA",
    "NEWS_SCHEMA",
    "TRAINING_SCHEMA",
    "build_schema",
    "coerce_to_contract",
    "validate_dataframe",
]

_log = get_logger(__name__)

#: Logical dtype to pandas dtype.
_DTYPE_MAP: Final[dict[ScalarDType, str]] = {
    "float": "float64",
    "int": "int64",
    "str": "str",
    "bool": "bool",
    "datetime": "datetime64[ns]",
    "category": "str",
}


def _checks_for(spec: FieldSpec) -> list[pa.Check]:
    """Build the pandera checks implied by a field specification.

    Args:
        spec: The field specification.

    Returns:
        The checks to attach to the column.
    """
    checks: list[pa.Check] = []
    if spec.ge is not None:
        checks.append(
            pa.Check.ge(spec.ge, name=f"{spec.name}_ge", error=f"{spec.name} < {spec.ge}")
        )
    if spec.le is not None:
        checks.append(
            pa.Check.le(spec.le, name=f"{spec.name}_le", error=f"{spec.name} > {spec.le}")
        )
    if spec.allowed is not None:
        checks.append(
            pa.Check.isin(
                list(spec.allowed),
                name=f"{spec.name}_allowed",
                error=f"{spec.name} outside the permitted set",
            )
        )
    return checks


def _column_for(spec: FieldSpec) -> pa.Column:
    """Compile one field specification into a pandera column.

    Args:
        spec: The field specification.

    Returns:
        The corresponding pandera column.
    """
    return pa.Column(
        dtype=_DTYPE_MAP[spec.dtype],
        checks=_checks_for(spec),
        nullable=spec.nullable,
        coerce=True,
        required=True,
        description=spec.description,
    )


def build_schema(
    contract: FeatureContract = FEATURE_CONTRACT,
    *,
    require_target: bool = True,
    require_identity: bool = True,
    strict: bool = False,
    unique_key: bool = True,
) -> pa.DataFrameSchema:
    """Compile a feature contract into a pandera schema.

    Args:
        contract: The contract to compile.
        require_target: Include the label column. Set false for inference, where
            the outcome is by definition unknown.
        require_identity: Include the identifier and date columns.
        strict: Reject columns absent from the contract. Left false by default
            so engineered intermediates can ride along without failing the batch.
        unique_key: Require ``(entity_id, as_of_date)`` to be unique, which is
            what stops a duplicated observation being counted twice.

    Returns:
        The compiled schema.
    """
    identity_columns = {ENTITY_ID_COLUMN, DATE_COLUMN}
    columns: dict[str, pa.Column] = {}

    for spec in contract.fields:
        if spec.name == TARGET_COLUMN and not require_target:
            continue
        if spec.name in identity_columns and not require_identity:
            continue
        columns[spec.name] = _column_for(spec)

    unique: list[str] | None = None
    if unique_key and require_identity:
        unique = [ENTITY_ID_COLUMN, DATE_COLUMN]

    return pa.DataFrameSchema(
        columns=columns,
        strict=strict,
        coerce=True,
        unique=unique,
        name="finml_modelling_dataset",
        description="Compiled from domain.entities.FEATURE_CONTRACT",
    )


#: Contract for a labelled training dataset.
TRAINING_SCHEMA: Final[pa.DataFrameSchema] = build_schema(require_target=True)

#: Contract for a scoring batch, where the label is absent by construction.
INFERENCE_SCHEMA: Final[pa.DataFrameSchema] = build_schema(
    require_target=False, require_identity=False, unique_key=False
)

#: Contract for the macro time series consumed by the satellite models.
MACRO_SCHEMA: Final[pa.DataFrameSchema] = pa.DataFrameSchema(
    columns={
        DATE_COLUMN: pa.Column("datetime64[ns]", coerce=True, nullable=False),
        **{
            name: pa.Column(
                "float64",
                checks=_checks_for(FEATURE_CONTRACT.get(name)),
                coerce=True,
                nullable=False,
            )
            for name in MACRO_VARIABLES
        },
    },
    strict=False,
    coerce=True,
    unique=[DATE_COLUMN],
    name="finml_macro_series",
)

#: Contract for the raw news corpus feeding the NLP stage.
NEWS_SCHEMA: Final[pa.DataFrameSchema] = pa.DataFrameSchema(
    columns={
        ENTITY_ID_COLUMN: pa.Column("str", coerce=True, nullable=False),
        "published_at": pa.Column("datetime64[ns]", coerce=True, nullable=False),
        "headline": pa.Column("str", coerce=True, nullable=False),
        "body": pa.Column("str", coerce=True, nullable=True),
    },
    strict=False,
    coerce=True,
    name="finml_news_corpus",
)


def validate_dataframe(
    df: pd.DataFrame,
    schema: pa.DataFrameSchema,
    *,
    context: str = "dataset",
) -> pd.DataFrame:
    """Validate a dataframe against a contract.

    Args:
        df: The dataframe to validate.
        schema: The contract to enforce.
        context: Label used in log events and the raised error, so a failure
            names which stage produced the bad data.

    Returns:
        The validated dataframe, with dtypes coerced to the contract.

    Raises:
        SchemaValidationError: If the dataframe violates the contract. The error
            context carries every individual failure, not just the first.
    """
    if df.empty:
        raise SchemaValidationError(
            "Refusing to validate an empty dataframe", context=context, schema=schema.name
        )
    try:
        validated = schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        failures = _summarise_failures(exc)
        _log.error(
            "schema.validation_failed",
            context=context,
            schema=schema.name,
            n_failures=len(failures),
            failures=failures[:20],
        )
        raise SchemaValidationError(
            "Dataframe violates its data contract",
            context=context,
            schema=schema.name,
            n_failure_cases=len(failures),
            failures=failures[:20],
        ) from exc

    _log.debug(
        "schema.validated",
        context=context,
        schema=schema.name,
        n_rows=len(validated),
        n_columns=len(validated.columns),
    )
    return validated


def _summarise_failures(exc: pa.errors.SchemaErrors) -> list[dict[str, Any]]:
    """Flatten a pandera failure report into loggable records.

    Args:
        exc: The raised pandera error.

    Returns:
        One record per failure case; a single fallback record if the report
        cannot be parsed.
    """
    try:
        cases = exc.failure_cases
        wanted = [c for c in ("schema_context", "column", "check", "failure_case") if c in cases]
        records = cases[wanted].drop_duplicates().to_dict(orient="records")
        return [{str(k): _stringify(v) for k, v in rec.items()} for rec in records]
    except (AttributeError, KeyError, TypeError) as parse_error:
        # Never let error *reporting* mask the original validation error.
        return [{"unparsed_report": str(exc)[:500], "parse_error": str(parse_error)}]


def _stringify(value: Any) -> str | float | int | bool | None:
    """Render a failure-case value as something JSON-serialisable.

    Args:
        value: The raw value from the pandera report.

    Returns:
        The value if it is already a JSON scalar, otherwise its string form.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def coerce_to_contract(
    df: pd.DataFrame,
    contract: FeatureContract = FEATURE_CONTRACT,
    *,
    fill_missing: bool = True,
) -> pd.DataFrame:
    """Reshape a dataframe to match the contract's feature space.

    Applied before scoring so a caller that omits an optional column, or sends
    columns in a different order, still gets a well-formed matrix. Column order
    matters: tree models index features positionally once fitted.

    Args:
        df: Input dataframe.
        contract: Contract describing the expected feature space.
        fill_missing: Create absent feature columns from the contract's
            ``typical`` value. When false, an absent column raises.

    Returns:
        A dataframe carrying exactly the contract's feature columns, in order.

    Raises:
        SchemaValidationError: If a feature is missing and ``fill_missing`` is
            false.
    """
    out = df.copy()
    missing: list[str] = []

    for spec in contract.fields:
        if not spec.is_feature or spec.name in out.columns:
            continue
        missing.append(spec.name)
        if not fill_missing:
            continue
        out[spec.name] = spec.allowed[0] if spec.allowed else spec.typical

    if missing and not fill_missing:
        raise SchemaValidationError(
            "Input is missing required feature columns", missing=missing
        )
    if missing:
        _log.warning(
            "schema.imputed_missing_columns",
            missing=missing,
            n_missing=len(missing),
            strategy="contract_typical",
        )

    return out[list(contract.feature_names)]
