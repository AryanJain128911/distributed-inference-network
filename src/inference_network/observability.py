"""JSON application logs with an explicit allowlist of non-sensitive fields."""

import json
import logging

EVENT_FIELDS = ("request_id", "provider_id", "attempt", "succeeded", "elapsed_ms", "error_type")


class EventFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = {"level": record.levelname, "event": record.getMessage()}
        for field in EVENT_FIELDS:
            if hasattr(record, field):
                event[field] = getattr(record, field)
        return json.dumps(event)


def configure_logging() -> None:
    logger = logging.getLogger("inference_network")
    handler = logging.StreamHandler()
    handler.setFormatter(EventFormatter())
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
