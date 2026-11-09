# Local inference traffic demos — HAProxy / HAPEE 3.2

Two independent demos for **Sovereign AI Starts at the Traffic Layer**:

| Demo | Listen | What the audience sees |
|---|---|---|
| [Model routing](model-routing/haproxy.cfg) | `127.0.0.1:18080` | JSON `model` selects an internal model-serving pool |
| [Consistent hashing](consistent-hashing/haproxy.cfg) | `127.0.0.1:18081` | A conversation stays on one replica; only its affected keys move when a replica fails |

No GPU, containers, model downloads, API keys, or Python packages are needed.
All listeners use loopback; the mocks make no outbound calls. The backend names
are illustrative GPU replicas, not actual accelerators.

**These mechanisms are native HAProxy features**, also available in HAPEE 3.2.
No Enterprise module is loaded or needed. This demonstrates capabilities of the
HAProxy data plane, not Enterprise-exclusive AI functionality.

## 1. Setup and automatic verification

Use Bash and Python 3. Set the absolute examples directory for your checkout
once, in each terminal where you run commands. Nothing depends on your working
directory:

```bash
DEMO=/absolute/path/to/your/checkout/examples
HAPROXY=/opt/hapee-3.2/sbin/hapee-lb

"$HAPROXY" -c -f "$DEMO/model-routing/haproxy.cfg"
"$HAPROXY" -c -f "$DEMO/consistent-hashing/haproxy.cfg"
HAPROXY_BIN="$HAPROXY" python3 "$DEMO/test_demos.py" -v
```

The tests run the actual binary against local mocks, substitute temporary free
ports into copies of the configs, check routing and rejection cases, stop one
replica, measure redistribution, and terminate only their own child processes.
They can run while the manual demos are up. A configuration error or occupied
port causes a test failure, not an attempt to stop another service.

A successful run reports 12 tests and a line like:

```text
Failover: 31/120 conversations moved; surviving assignments stable.
```

That number is an observation for this configuration and key sample, not a
promise about every workload or hash-ring change.

## 2. Model routing

### Terminal A: start the two model servers and HAProxy

The manual demo needs ports 18080, 18101 and 18102 free. Don't start it twice.
The parenthesized Bash block keeps its cleanup trap out of your interactive
shell. Ctrl-C stops HAProxy and cleans up the remaining background mock jobs.

```bash
(
    set -eu
    trap 'for pid in $(jobs -pr); do kill "$pid" 2>/dev/null || true; done; wait || true' EXIT
    python3 "$DEMO/mock_inference.py" --port 18101 --model llama-3.1-8b --replica llama-1 &
    python3 "$DEMO/mock_inference.py" --port 18102 --model qwen2.5-7b --replica qwen-1 &
    "$HAPROXY" -db -f "$DEMO/model-routing/haproxy.cfg"
)
```

Wait for the mock startup messages before sending requests. Health checks run
every 500 ms. A failed startup prints an error; stop the block and resolve it
rather than using an unrelated listener that happens to occupy a demo port.

### Terminal B: select models through the same endpoint

```bash
curl -si http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama-3.1-8b","messages":[{"role":"user","content":"Hello"}]}'

curl -si http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-7b","messages":[{"role":"user","content":"Hello"}]}'
```

Both return `200`, a JSON `chat.completion`, and respectively
`x-mock-replica: llama-1` or `x-mock-replica: qwen-1`. Header names are
case-insensitive. `model` is an exact, case-sensitive served-model alias, as an
inference server can expose with its served-model-name setting.

Try changing the model to `unknown`, omitting it, or using a number: the gateway
returns `400` instead of choosing a default pool. No external fallback exists.
If a pool's only replica is down, its model becomes unavailable (`503`) rather
than silently switching to a different model.

### The lines to explain

```haproxy
http-request wait-for-body time 2s
# ... check that the complete bounded body is available ...
http-request set-var(txn.model) req.body,json_query('$.model')
acl llama var(txn.model) -m str llama-3.1-8b
use_backend llama_pool if llama
```

Each pool has one mock to keep model selection distinct from replica selection.
A real pool can contain multiple replicas of the same served model.

## 3. Consistent hashing across three replicas

### Terminal A (or C): start the replicas and HAProxy

This demo uses ports 18081 and 18201–18203, so it can run beside model routing.

```bash
(
    set -eu
    trap 'for pid in $(jobs -pr); do kill "$pid" 2>/dev/null || true; done; wait || true' EXIT
    python3 "$DEMO/mock_inference.py" --port 18201 --model llama-3.1-8b --replica gpu-1 &
    python3 "$DEMO/mock_inference.py" --port 18202 --model llama-3.1-8b --replica gpu-2 &
    gpu2=$!
    python3 "$DEMO/mock_inference.py" --port 18203 --model llama-3.1-8b --replica gpu-3 &
    printf '\nFor the failover step, stop only this gpu-2 process: kill %s\n\n' "$gpu2"
    "$HAPROXY" -db -f "$DEMO/consistent-hashing/haproxy.cfg"
)
```

### Terminal B: repeat a conversation

```bash
for i in 1 2 3; do
    curl -si http://127.0.0.1:18081/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -H 'X-Conversation-ID: conversation-42' \
      -d '{"model":"llama-3.1-8b","messages":[{"role":"user","content":"Hello"}]}'
done
```

The replica header stays the same for that key while the pool is healthy and
unchanged. Other keys may land on other replicas; two distinct keys need not
land on distinct replicas. Missing or blank conversation IDs return `400`;
multiple header values are rejected as ambiguous.

For a compact view of several assignments:

```bash
for i in $(seq 0 11); do
    printf 'conversation-%s ' "$i"
    curl -sS -D - -o /dev/null http://127.0.0.1:18081/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -H "X-Conversation-ID: conversation-$i" \
      -d '{"model":"llama-3.1-8b","messages":[{"role":"user","content":"Hello"}]}' \
      | tr -d '\r' | awk 'tolower($1) == "x-mock-replica:" {print $2}'
done
```

Run **only the `kill PID` command printed by your launch block**, in another
terminal, to stop its `gpu-2`. Wait about two seconds for health checks, then
repeat the sample. Keys previously on `gpu-2` move to a survivor; the other
assignments stay stable in this configuration. Before health checks notice,
`option redispatch` sends a refused connection to a surviving replica instead
of failing it. A request already in flight to the dying replica still fails;
hashing does not make an in-flight inference survive.
The automatic test checks this over 120 keys without manual PID handling.

To restore the replica while HAProxy is still running, use another terminal:

```bash
python3 "$DEMO/mock_inference.py" --port 18202 --model llama-3.1-8b --replica gpu-2
```

After a successful health check the replica rejoins the ring. Stop this manually
restarted process separately with Ctrl-C; it is not owned by the original block.

### The lines to explain

```haproxy
balance hdr(X-Conversation-ID)
hash-type consistent
retries 2
option redispatch 1
```

No stick table or per-conversation persistence state is required. Each server
has an explicit stable `id` and equal weight. Keep identities, weights, hash
settings and pool membership consistent across gateway instances if they must
make the same choice. Consistent hashing limits movement; it neither guarantees
an even split for a small key sample nor reacts to GPU load or queue depth.

To compose both demos, keep model selection in the frontend and use consistent
hashing *inside each model-specific backend*. Do not hash across servers that
serve different models and hope the selected server can handle the request.

## Boundaries and claims

- **OpenAI-style subset:** `POST /v1/chat/completions` accepts a string `model`
  and a nonempty `messages` array with text `content` and roles `system`,
  `developer`, `user`, or `assistant`. `stream` must be absent or `false`.
  Other generation parameters are ignored. Responses contain completion IDs,
  timestamps, a single assistant choice, and `finish_reason: stop`; they do not
  fabricate token usage. There is no streaming, tools, multimodal input, model
  listing endpoint or inference. The text always explicitly says `MOCK`.
- **Body inspection:** model routing requires `Content-Length`, rejects chunked
  uploads (`411`), and limits bodies to 8192 bytes (`413`). Waiting for a body is
  bounded to two seconds (`408`). A 32768-byte buffer includes headers too;
  large headers can leave too little room even for an otherwise allowed body.
  The config rejects incomplete buffered bodies instead of routing from them.
  The mocks independently enforce the same 8192-byte size limit.
- **Extraction is not sanitization:** native `json_query` extracts a routing
  hint, not a schema-validation verdict. Duplicate/encoded JSON keys can have
  implementation-dependent extraction behavior. The mocks reject duplicate
  keys and malformed JSON, validate messages, and verify that the requested
  model matches the served model. Real inference servers must also validate
  their requests; never use this extraction alone as a security policy.
- **Affinity is not caching:** the conversation header is an explicit demo
  routing key, not an OpenAI API field or a hash of the prompt. Stable routing
  can help *if* the inference engine supports suitable cache reuse. The mocks
  have no KV/prefix cache or conversation state. A hot key can overload a
  replica; failover does not transfer GPU state or guarantee a cache hit.
- **Not a production gateway:** no auth, TLS, tenant isolation, rate limits,
  Kubernetes discovery, readiness integration, or retry/idempotency policy.
  Health checks use the mock's `/health`, which is not an OpenAI API standard.
  HAProxy-generated error bodies are not uniformly OpenAI-style errors.
- **Sovereignty and performance:** this demo's request path stays on the laptop.
  Cluster-only inference needs actual internal endpoints, network/egress policy,
  and verification of the inference service's own behavior. Body buffering and
  parsing have costs; these tests do **not** establish “no performance penalty.”
