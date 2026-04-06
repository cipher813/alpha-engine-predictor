"""
Load secrets from AWS SSM Parameter Store into os.environ.

All Alpha Engine secrets are stored under the /alpha-engine/ prefix in SSM
Parameter Store (SecureString type). This module fetches them at startup and
sets os.environ so existing code (which uses os.environ.get()) works unchanged.

Falls back to .env file if SSM is unavailable (local development).

Usage (call once at module startup, before any os.environ.get):
    from ssm_secrets import load_secrets
    load_secrets()
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

SSM_PREFIX = "/alpha-engine/"
_loaded = False


def load_secrets(prefix: str = SSM_PREFIX, region: str | None = None) -> int:
    """
    Fetch all parameters under prefix from SSM and set as env vars.

    Parameter names are converted to env var names by stripping the prefix
    and converting to uppercase. E.g., /alpha-engine/POLYGON_API_KEY → POLYGON_API_KEY.

    Returns the number of parameters loaded. Skips any that are already set
    in the environment (explicit env vars take precedence over SSM).

    Falls back silently if SSM is unavailable (e.g., local dev without AWS creds).
    """
    global _loaded
    if _loaded:
        return 0

    region = region or os.environ.get("AWS_REGION", "us-east-1")
    count = 0

    try:
        import boto3
        client = boto3.client("ssm", region_name=region)

        paginator = client.get_paginator("get_parameters_by_path")
        pages = paginator.paginate(
            Path=prefix,
            Recursive=False,
            WithDecryption=True,
        )

        for page in pages:
            for param in page.get("Parameters", []):
                name = param["Name"]
                value = param["Value"]
                env_key = name.replace(prefix, "", 1)
                if not env_key:
                    continue
                if env_key not in os.environ:
                    os.environ[env_key] = value
                    count += 1
                else:
                    logger.debug("SSM skip %s: already set in environment", env_key)

        _loaded = True
        logger.info("Loaded %d secrets from SSM %s", count, prefix)

    except ImportError:
        logger.debug("boto3 not available — skipping SSM secrets load")
    except Exception as e:
        logger.warning("SSM secrets load failed (falling back to env): %s", e)

    return count
