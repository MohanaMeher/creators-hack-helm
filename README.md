# Helm - Real Agent Runtime Control Plane

A production-grade hierarchical control plane for autonomous AI agents using Claude (Anthropic) and Tonic for synthetic data generation.

## Features

- **Real Worker Agents**: Classifier and Processor agents that perform actual work using Claude API
- **Real Telemetry**: All metrics derived from actual API calls and execution
- **Autonomous Helm Governors**: Agent-level and service-level governors that run continuously
- **Real Drift Detection**: Rolling baselines computed from actual telemetry
- **Live Observability**: Real-time charts and metrics via WebSocket
- **Hierarchical Governance**: Agent → Service escalation with autonomous recovery

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```bash
# Required: Anthropic API Key for Claude
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: Tonic API (currently using built-in generator)
TONIC_API_KEY=your_tonic_api_key_here
TONIC_API_URL=https://api.tonic.ai/v1
```

**You need to provide your Anthropic API key** - get it from https://console.anthropic.com/

### 3. Run the Application

```bash
python main.py
```

Or with uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### 4. Access the Control Plane

Open http://localhost:8001 in your browser.

## Architecture

Helm is designed as a **control-plane system for governing AI agents**, not as an agent itself.

It separates **execution** (doing work) from **governance** (monitoring and intervention), following the same architectural pattern used by systems like Kubernetes and modern infra control planes.

---

### High-level model

In production, Helm would operate as a **control-plane SaaS**, while AI agents run in a **customer-owned execution environment**.

For demo simplicity, both planes are colocated in a single FastAPI process.

```
Execution Plane (Customer)
--------------------------
• Worker Agents
• Task execution
• LLM calls
• Data processing
• Telemetry emission


Control Plane (Helm)
-------------------
• Observer agents
• Governance policies
• Agent-level governors
• Service-level governors
• State & incident tracking
• Control-plane UI
```

---

### Execution plane (worker agents)

Worker agents are **real autonomous agents** that:

* perform actual work (e.g., classification, processing)
* call external models (Claude)
* operate on realistic data (via Tonic)
* retry on failure
* emit real telemetry

They are **inside Helm's system boundary** but **not part of the control plane**.

Their responsibility is execution, not governance.

**Real Worker Agents:**

1. **Classifier Agent**: Uses Claude to classify customer records into risk categories
   - Processes Tonic-generated customer data
   - Emits real telemetry from API calls
   - Supports anomaly injection (malformed data, increased batch size)

2. **Processor Agent**: Uses Claude to summarize incident reports
   - Processes Tonic-generated incident reports
   - Real token usage and confidence scores

---

### Control plane (Helm)

Helm is the **system that governs agents**, not a single agent.

It consists of multiple autonomous control components:

#### Observer Agent

* Continuously analyzes telemetry
* Computes rolling baselines (median of last 20 values)
* Detects behavioral drift:
  - Token usage > 5× rolling baseline
  - Confidence < 0.55 for 3+ consecutive runs
  - Repeated retries (3+ runs with retries)
* Emits drift events

#### Agent-Level Helm Governors

* One per worker agent
* Runs continuously as autonomous agents
* Pause agents on drift
* Apply tighter operating constraints (rate limits, batch size, max retries)
* Attempt recovery

#### Service-Level Helm Governor

* Governs a group of agents
* Receives escalations from agent-level governors
* Contains blast radius when recovery fails
* Enforces service-wide limits

Governance is **hierarchical**:

```
Agent → Service → Organization
```

No single "god agent" exists.

---

### State model

All state is kept in memory for the demo:

* Helm hierarchy (org → service → agents)
* Agent state: `healthy | paused | degraded`
* Incident log (append-only, time-ordered)
* Live telemetry

This mirrors how a real control plane would behave, without external dependencies.

---

### UI (control-plane view)

The UI represents what an **operator or platform team** would see:

* Current agent hierarchy and states
* Live agent metrics
* Recent interventions and escalations
* Time-series charts for token usage and confidence
* Interactive chat to talk directly to agents

It is intentionally minimal and utilitarian, reflecting a real internal infra dashboard rather than a product UI.

---

### Real Anomaly Injection

The "Simulate anomaly" button causes:
- Increased batch size (more tokens per run)
- Malformed data injection (30% chance)
- Real API failures and retries
- Observable Helm intervention

---

### Demo vs production

For this demo:

* Execution plane and control plane are colocated
* Communication is in-memory
* Telemetry is streamed via WebSocket

In production:

* Worker agents would run in customer environments
* Helm would operate remotely as a control-plane SaaS
* Communication would occur via APIs/events

The **architecture and governance model remain the same**.

---

### Design principle

> Agents do work.
> Helm governs work.

This separation is what allows autonomy to scale safely.

## API Endpoints

- `GET /` - Control plane UI with chat interface
- `GET /state` - Complete Helm hierarchy and state
- `POST /simulate/anomaly/{agent_id}` - Inject real anomalies
- `POST /agent/{agent_id}/query` - Send queries to agents (chat interface)
- `WS /ws` - WebSocket for real-time updates

## Notes

- All agents run as long-lived background threads
- Telemetry is real (from Claude API responses)
- Charts update in real-time via WebSocket
- No simulation - this is a real agentic system
