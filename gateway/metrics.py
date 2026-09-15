"""Gateway metrics, served at /__capture/metrics.

Separate process from the agentmemory API, so the default registry is ours
alone and there is no collision with app/metrics.py.
"""
from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("capture_requests_total", "Proxied requests",
                   ["endpoint", "streamed"])
UPSTREAM_ERRORS = Counter("capture_upstream_errors_total",
                          "Upstream transport failures", ["endpoint"])
# Endpoints that pass through with no capture adapter. Nothing is meant to
# use these; the counter is here so that if something starts, it shows up
# rather than being silently missed.
UNCAPTURED = Counter("capture_uncaptured_total",
                     "Proxied requests on endpoints with no capture adapter",
                     ["endpoint"])
TRUNCATED = Counter("capture_truncated_total",
                    "Streams the client hung up on mid-flight")
TOOL_STEPS = Counter("gateway_tool_steps_total",
                     "Model+tool round trips run inside agent loops")
COMPACTED = Counter("capture_compacted_total",
                    "Requests whose middle was summarised to fit the context")
# Machine traffic that identified itself as write-nothing. No transcript
# exists for these, so this counter is the only record that they happened —
# which is the point: unarchived is not the same as unaccounted for.
SUPPRESSED = Counter("capture_suppressed_total",
                     "Exchanges proxied without writing a transcript",
                     ["client"])
# Requests whose client-supplied sampling fields were dropped so the model
# server's own settings apply (GATEWAY_STRIP_SAMPLING).
SAMPLING_STRIPPED = Counter("gateway_sampling_stripped_total",
                            "Requests forwarded without the client's sampling fields")
DROPPED = Counter("capture_dropped_total", "Capture records dropped",
                  ["reason"])
FLUSHES = Counter("capture_flush_total", "Conversation flushes", ["outcome"])
ADOPTED = Counter("capture_adopted_total", "Requests matched to a conversation",
                  ["how"])

OPEN_CONVS = Gauge("capture_open_conversations", "Conversations awaiting flush")
QUEUE_DEPTH = Gauge("capture_queue_depth", "Capture records awaiting apply")
PENDING_RETRY = Gauge("capture_pending_retry", "Records waiting on vault retry")
VAULT_UP = Gauge("capture_vault_up", "1 when the last vault write succeeded")

FLUSH_LAT = Histogram("capture_flush_seconds",
                      "Render + write time for one transcript")
