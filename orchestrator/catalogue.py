from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from .config import Config

log = logging.getLogger("orchestrator.catalogue")

_BLOCK_TYPES = [
    "tools", "topics", "commands", "procedures",
    "components", "memory", "selection", "filter", "devices",
]


class CatalogueService:
    """Loads the saved-block catalogue for a project from AWS S3."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._s3_client: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def load(self, user_id: str, project_id: str, username: str = "") -> str:
        """Return a formatted catalogue string, or a fallback message on error."""
        s3 = self._get_s3_client()
        if not s3 or not self._config.s3_bucket:
            return "(S3 catalogue not available)"

        prefix = (
            f"users/{user_id}/{username}/{project_id}/"
            if username
            else f"{user_id}/{project_id}/"
        )

        loop = asyncio.get_event_loop()
        try:
            entries = await loop.run_in_executor(
                None, self._list_and_load_blocks, s3, prefix
            )
        except Exception as exc:
            log.warning("S3 executor error: %s", exc)
            return "(S3 catalogue load error)"

        if not entries:
            return "(No saved blocks found in this project)"
        return f"SAVED BLOCKS CATALOGUE ({len(entries)} blocks):\n" + "\n".join(entries)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_s3_client(self) -> Any:
        """Lazy-initialise the boto3 S3 client when AWS credentials are present."""
        if self._s3_client is None and (
            os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE")
        ):
            try:
                self._s3_client = boto3.client(
                    "s3", region_name=self._config.aws_region
                )
            except Exception as exc:
                log.warning("Could not create S3 client: %s", exc)
        return self._s3_client

    def _list_and_load_blocks(self, s3: Any, prefix: str) -> list[str]:
        """Synchronous S3 scan — intended to run in a thread-pool executor."""
        results: list[str] = []
        try:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._config.s3_bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    entry = self._load_block_entry(s3, obj["Key"], prefix)
                    if entry:
                        results.append(entry)
        except (ClientError, BotoCoreError) as exc:
            log.warning("S3 catalogue load failed: %s", exc)
        return results

    def _load_block_entry(self, s3: Any, key: str, prefix: str) -> str | None:
        """Parse a single S3 object key and return a catalogue line, or None to skip."""
        parts = key[len(prefix):].split("/")
        # Expected structure: {block_type}/{block_name}/{block_name}.json
        if len(parts) < 3 or parts[0] not in _BLOCK_TYPES or not parts[-1].endswith(".json"):
            return None

        block_type = parts[0]
        block_name = parts[-1][:-5]  # strip .json

        line = f'  {block_name} | block_type="{block_type}" block_name="{block_name}"'
        try:
            resp = s3.get_object(Bucket=self._config.s3_bucket, Key=key)
            data = json.loads(resp["Body"].read())
            desc = data.get("description", "")
            if not desc:
                tool_calls = data.get("tool_calls", [])
                if tool_calls:
                    desc = tool_calls[0].get("params", {}).get("description", "")
            if desc:
                line += f" — {desc[:80]}"
        except Exception:
            pass

        return line
