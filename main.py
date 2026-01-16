"""
Helm - Hierarchical Control Plane for AI Agents
A single-process FastAPI application simulating agent governance.
"""
import asyncio
import json
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse


# ============================================================================
# Data Models
# ============================================================================

class AgentState(str, Enum):
    HEALTHY = "healthy"
    PAUSED = "paused"
    DEGRADED = "degraded"


@dataclass
class Telemetry:
    """Agent telemetry data."""
    agent_id: str
    timestamp: float
    runs_per_min: float
    tokens_per_run: float
    confidence_score: float
    retry_count: int


@dataclass
class Incident:
    """Incident log entry."""
    timestamp: float
    agent_id: str
    message: str
    severity: str


@dataclass
class AgentInfo:
    """Agent information and state."""
    agent_id: str
    name: str
    state: AgentState
    current_telemetry: Optional[Telemetry] = None
    baseline_tokens: float = 0.0
    bad_behavior_mode: bool = False


@dataclass
class HelmState:
    """Complete Helm hierarchy and state."""
    org: str
    services: Dict[str, Dict[str, AgentInfo]]
    incidents: List[Incident]
    last_intervention: Optional[str] = None


# ============================================================================
# Worker Agents
# ============================================================================

class WorkerAgent:
    """Simulates a worker agent that emits telemetry."""
    
    def __init__(self, agent_id: str, name: str, baseline_tokens: float = 100.0):
        self.agent_id = agent_id
        self.name = name
        self.baseline_tokens = baseline_tokens
        self.bad_behavior_mode = False
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.telemetry_queue: deque = deque(maxlen=60)  # Keep last 60 seconds
        
    def start(self):
        """Start the agent in a background thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop the agent."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            
    def set_bad_behavior(self, enabled: bool):
        """Enable or disable bad behavior mode."""
        self.bad_behavior_mode = enabled
        
    def _run(self):
        """Main agent loop - emits telemetry every second."""
        while self.running:
            telemetry = self._generate_telemetry()
            self.telemetry_queue.append(telemetry)
            
            # Emit to global state
            if self.agent_id in app_state.agents:
                app_state.agents[self.agent_id].current_telemetry = telemetry
                app_state.broadcast_telemetry(telemetry)
            
            time.sleep(1.0)
            
    def _generate_telemetry(self) -> Telemetry:
        """Generate telemetry data."""
        if self.bad_behavior_mode:
            # Bad behavior: high tokens, low confidence, high retries
            tokens = self.baseline_tokens * 6.0  # > 5× baseline
            confidence = 0.50  # < 0.55
            retries = 5
            runs = 10.0
        else:
            # Normal behavior
            tokens = self.baseline_tokens * (0.9 + 0.2 * (time.time() % 1.0))
            confidence = 0.85 + 0.1 * (time.time() % 1.0)
            retries = 0
            runs = 20.0 + 5.0 * (time.time() % 1.0)
            
        return Telemetry(
            agent_id=self.agent_id,
            timestamp=time.time(),
            runs_per_min=runs,
            tokens_per_run=tokens,
            confidence_score=confidence,
            retry_count=retries
        )
    
    def get_recent_telemetry(self, seconds: int = 10) -> List[Telemetry]:
        """Get telemetry from the last N seconds."""
        cutoff = time.time() - seconds
        return [t for t in self.telemetry_queue if t.timestamp >= cutoff]


# ============================================================================
# Observer & Drift Detection
# ============================================================================

class Observer:
    """Observes agent telemetry and detects drift."""
    
    def __init__(self):
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start the observer loop."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop the observer."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            
    def _run(self):
        """Main observer loop - checks for drift every 2 seconds."""
        while self.running:
            for agent_id, agent_info in app_state.agents.items():
                if agent_info.state == AgentState.PAUSED:
                    continue
                    
                worker = app_state.worker_agents.get(agent_id)
                if not worker:
                    continue
                    
                recent_telemetry = worker.get_recent_telemetry(seconds=10)
                if not recent_telemetry:
                    continue
                    
                drift_detected = self._detect_drift(agent_info, recent_telemetry)
                if drift_detected:
                    app_state.handle_drift(agent_id, drift_detected)
                    
            time.sleep(2.0)
            
    def _detect_drift(self, agent_info: AgentInfo, telemetry: List[Telemetry]) -> Optional[str]:
        """Detect drift conditions. Returns drift reason or None."""
        if not telemetry:
            return None
            
        latest = telemetry[-1]
        baseline = agent_info.baseline_tokens
        
        # Condition 1: Token usage > 5× baseline
        if latest.tokens_per_run > baseline * 5.0:
            return f"Token usage {latest.tokens_per_run:.1f} exceeds 5× baseline {baseline:.1f}"
            
        # Condition 2: Confidence < 0.55
        if latest.confidence_score < 0.55:
            return f"Confidence {latest.confidence_score:.2f} below threshold 0.55"
            
        # Condition 3: Retry loop detected (retries > 3)
        if latest.retry_count > 3:
            return f"Retry loop detected: {latest.retry_count} retries"
            
        return None


# ============================================================================
# Helm Governors
# ============================================================================

class AgentHelm:
    """Agent-level Helm governor - governs a single agent."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        
    def handle_drift(self, reason: str) -> bool:
        """Handle drift for this agent. Returns True if recovered, False if needs escalation."""
        agent_info = app_state.agents.get(self.agent_id)
        if not agent_info:
            return False
            
        # Pause the agent
        if agent_info.state != AgentState.PAUSED:
            agent_info.state = AgentState.PAUSED
            app_state.log_incident(
                self.agent_id,
                f"Agent paused due to drift: {reason}",
                "warning"
            )
            app_state.last_intervention = f"Agent {agent_info.name} paused by AgentHelm"
            app_state.broadcast_state_change()
            
        # Attempt recovery
        self.recovery_attempts += 1
        if self.recovery_attempts <= self.max_recovery_attempts:
            # Try to recover by disabling bad behavior mode
            worker = app_state.worker_agents.get(self.agent_id)
            if worker and worker.bad_behavior_mode:
                worker.set_bad_behavior(False)
                time.sleep(2.0)  # Wait a bit
                
            # Check if recovered
            worker = app_state.worker_agents.get(self.agent_id)
            if worker:
                recent = worker.get_recent_telemetry(seconds=5)
                if recent and not self._still_drifting(recent[-1], agent_info):
                    attempt_num = self.recovery_attempts
                    agent_info.state = AgentState.HEALTHY
                    self.recovery_attempts = 0
                    app_state.log_incident(
                        self.agent_id,
                        f"Agent recovered successfully (attempt {attempt_num})",
                        "info"
                    )
                    app_state.last_intervention = f"Agent {agent_info.name} recovered by AgentHelm"
                    app_state.broadcast_state_change()
                    return True
                    
        # Recovery failed - needs escalation
        agent_info.state = AgentState.DEGRADED
        app_state.log_incident(
            self.agent_id,
            f"Agent recovery failed after {self.recovery_attempts} attempts. Escalating to ServiceHelm.",
            "error"
        )
        app_state.last_intervention = f"Agent {agent_info.name} escalated to ServiceHelm"
        app_state.broadcast_state_change()
        return False
        
    def _still_drifting(self, telemetry: Telemetry, agent_info: AgentInfo) -> bool:
        """Check if agent is still drifting."""
        baseline = agent_info.baseline_tokens
        return (telemetry.tokens_per_run > baseline * 5.0 or
                telemetry.confidence_score < 0.55 or
                telemetry.retry_count > 3)


class ServiceHelm:
    """Service-level Helm governor - governs multiple agent-level Helms."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.agent_helms: Dict[str, AgentHelm] = {}
        
    def register_agent(self, agent_id: str):
        """Register an agent under this service."""
        self.agent_helms[agent_id] = AgentHelm(agent_id)
        
    def handle_escalation(self, agent_id: str):
        """Handle escalation from an agent-level Helm."""
        agent_info = app_state.agents.get(agent_id)
        if not agent_info:
            return
            
        # Service-level intervention
        app_state.log_incident(
            agent_id,
            f"ServiceHelm intervention for {agent_info.name} in service {self.service_name}",
            "critical"
        )
        app_state.last_intervention = f"ServiceHelm intervention for {agent_info.name}"
        app_state.broadcast_state_change()
        
        # Service-level recovery attempt: force reset
        worker = app_state.worker_agents.get(agent_id)
        if worker:
            worker.set_bad_behavior(False)
            time.sleep(3.0)
            
            # Check recovery
            recent = worker.get_recent_telemetry(seconds=5)
            if recent:
                latest = recent[-1]
                baseline = agent_info.baseline_tokens
                if (latest.tokens_per_run <= baseline * 5.0 and
                    latest.confidence_score >= 0.55 and
                    latest.retry_count <= 3):
                    agent_info.state = AgentState.HEALTHY
                    app_state.log_incident(
                        agent_id,
                        f"ServiceHelm successfully recovered {agent_info.name}",
                        "info"
                    )
                    app_state.last_intervention = f"ServiceHelm recovered {agent_info.name}"
                    app_state.broadcast_state_change()
                else:
                    app_state.log_incident(
                        agent_id,
                        f"ServiceHelm recovery failed for {agent_info.name}. Manual intervention required.",
                        "critical"
                    )


# ============================================================================
# Application State
# ============================================================================

class ApplicationState:
    """Global application state."""
    
    def __init__(self):
        self.org = "ai-org"
        self.services: Dict[str, Dict[str, AgentInfo]] = {}
        self.agents: Dict[str, AgentInfo] = {}
        self.worker_agents: Dict[str, WorkerAgent] = {}
        self.agent_helms: Dict[str, AgentHelm] = {}
        self.service_helms: Dict[str, ServiceHelm] = {}
        self.incidents: List[Incident] = []
        self.last_intervention: Optional[str] = None
        self.websocket_connections: Set[WebSocket] = set()
        self.observer: Optional[Observer] = None
        self.message_queue: queue.Queue = queue.Queue()
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        
    def initialize(self):
        """Initialize the application with default agents."""
        # Create service
        service_name = "ai-service"
        self.services[service_name] = {}
        
        # Create service-level Helm
        service_helm = ServiceHelm(service_name)
        self.service_helms[service_name] = service_helm
        
        # Create agents
        agents_config = [
            ("classifier", "Classifier Agent", 100.0),
            ("processor", "Processor Agent", 150.0),
        ]
        
        for agent_id, name, baseline_tokens in agents_config:
            agent_info = AgentInfo(
                agent_id=agent_id,
                name=name,
                state=AgentState.HEALTHY,
                baseline_tokens=baseline_tokens
            )
            self.agents[agent_id] = agent_info
            self.services[service_name][agent_id] = agent_info
            
            # Create worker agent
            worker = WorkerAgent(agent_id, name, baseline_tokens)
            self.worker_agents[agent_id] = worker
            worker.start()
            
            # Create agent-level Helm
            agent_helm = AgentHelm(agent_id)
            self.agent_helms[agent_id] = agent_helm
            service_helm.register_agent(agent_id)
            
        # Start observer
        self.observer = Observer()
        self.observer.start()
        
    def handle_drift(self, agent_id: str, reason: str):
        """Handle drift detection for an agent."""
        agent_helm = self.agent_helms.get(agent_id)
        if not agent_helm:
            return
            
        recovered = agent_helm.handle_drift(reason)
        if not recovered:
            # Escalate to service-level Helm
            agent_info = self.agents.get(agent_id)
            if agent_info:
                # Find which service this agent belongs to
                for service_name, agents in self.services.items():
                    if agent_id in agents:
                        service_helm = self.service_helms.get(service_name)
                        if service_helm:
                            service_helm.handle_escalation(agent_id)
                        break
                        
    def log_incident(self, agent_id: str, message: str, severity: str):
        """Log an incident."""
        incident = Incident(
            timestamp=time.time(),
            agent_id=agent_id,
            message=message,
            severity=severity
        )
        self.incidents.append(incident)
        self.broadcast_incident(incident)
        
    def get_helm_state(self) -> HelmState:
        """Get the complete Helm state."""
        return HelmState(
            org=self.org,
            services=self.services,
            incidents=self.incidents,
            last_intervention=self.last_intervention
        )
    
    def _serialize_helm_state(self, state: HelmState) -> dict:
        """Serialize HelmState to dict, handling nested structures."""
        result = {
            "org": state.org,
            "services": {},
            "incidents": [asdict(inc) for inc in state.incidents],
            "last_intervention": state.last_intervention
        }
        # Serialize nested services structure
        for service_name, agents in state.services.items():
            result["services"][service_name] = {
                agent_id: asdict(agent_info)
                for agent_id, agent_info in agents.items()
            }
        return result
        
    def broadcast_telemetry(self, telemetry: Telemetry):
        """Broadcast telemetry to all WebSocket connections."""
        message = {
            "type": "telemetry",
            "data": asdict(telemetry)
        }
        self._broadcast(message)
        
    def broadcast_state_change(self):
        """Broadcast state change to all WebSocket connections."""
        helm_state = self.get_helm_state()
        message = {
            "type": "state_change",
            "data": self._serialize_helm_state(helm_state)
        }
        self._broadcast(message)
        
    def broadcast_incident(self, incident: Incident):
        """Broadcast incident to all WebSocket connections."""
        message = {
            "type": "incident",
            "data": asdict(incident)
        }
        self._broadcast(message)
        
    def _broadcast(self, message: dict):
        """Broadcast message to all connected WebSockets via queue."""
        self.message_queue.put(message)


# Global application state
app_state = ApplicationState()


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(title="Helm - Agent Runtime Control Plane")


async def broadcast_worker():
    """Background task to process WebSocket broadcasts."""
    while True:
        try:
            # Check for messages (non-blocking)
            try:
                message = app_state.message_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.1)
                continue
                
            # Broadcast to all connections
            disconnected = set()
            for ws in app_state.websocket_connections:
                try:
                    await ws.send_json(message)
                except Exception:
                    disconnected.add(ws)
            app_state.websocket_connections -= disconnected
        except Exception as e:
            print(f"Error in broadcast worker: {e}")
            await asyncio.sleep(0.1)


@app.on_event("startup")
async def startup():
    """Initialize application on startup."""
    try:
        app_state.event_loop = asyncio.get_event_loop()
        app_state.initialize()
        asyncio.create_task(broadcast_worker())
        print("Helm application started successfully")
    except Exception as e:
        print(f"Error during startup: {e}")
        import traceback
        traceback.print_exc()


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    for worker in app_state.worker_agents.values():
        worker.stop()
    if app_state.observer:
        app_state.observer.stop()


@app.get("/")
async def get_ui():
    """Serve the control plane UI."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Helm - Agent Runtime Control Plane</title>
    <style>
        body {
            font-family: monospace;
            margin: 20px;
            background: #1a1a1a;
            color: #e0e0e0;
        }
        h1 {
            color: #4a9eff;
            border-bottom: 2px solid #4a9eff;
            padding-bottom: 10px;
        }
        h2 {
            color: #6ac259;
            margin-top: 30px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: #2a2a2a;
        }
        th, td {
            border: 1px solid #444;
            padding: 10px;
            text-align: left;
        }
        th {
            background: #333;
            color: #4a9eff;
        }
        tr:hover {
            background: #333;
        }
        .state-healthy { color: #6ac259; }
        .state-paused { color: #ffa500; }
        .state-degraded { color: #ff4444; }
        .indent-1 { padding-left: 30px; }
        .indent-2 { padding-left: 60px; }
        button {
            background: #4a9eff;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 14px;
            cursor: pointer;
            margin: 20px 0;
        }
        button:hover {
            background: #5aaeff;
        }
        .incident-log {
            max-height: 300px;
            overflow-y: auto;
            background: #2a2a2a;
            padding: 10px;
            border: 1px solid #444;
        }
        .incident {
            margin: 5px 0;
            padding: 5px;
            border-left: 3px solid #444;
        }
        .incident-info { border-left-color: #4a9eff; }
        .incident-warning { border-left-color: #ffa500; }
        .incident-error { border-left-color: #ff4444; }
        .incident-critical { border-left-color: #ff0000; }
    </style>
</head>
<body>
    <h1>Helm — Agent Runtime Control Plane</h1>
    
    <button onclick="simulateAnomaly()">Simulate anomaly on classifier</button>
    
    <h2>Hierarchy Table</h2>
    <table id="hierarchy-table">
        <thead>
            <tr>
                <th>Level</th>
                <th>Name</th>
                <th>State</th>
            </tr>
        </thead>
        <tbody id="hierarchy-body">
        </tbody>
    </table>
    
    <h2>Live Agent Metrics</h2>
    <table id="metrics-table">
        <thead>
            <tr>
                <th>Agent</th>
                <th>Runs/min</th>
                <th>Tokens/run</th>
                <th>Confidence</th>
                <th>Retries</th>
                <th>Timestamp</th>
            </tr>
        </thead>
        <tbody id="metrics-body">
        </tbody>
    </table>
    
    <h2>Last Intervention</h2>
    <div id="last-intervention" style="padding: 10px; background: #2a2a2a; border: 1px solid #444;">
        <em>No interventions yet</em>
    </div>
    
    <h2>Incident Log</h2>
    <div id="incident-log" class="incident-log">
        <em>No incidents yet</em>
    </div>
    
    <script>
        let ws = null;
        let currentState = null;
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                console.log('WebSocket connected');
            };
            
            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                handleMessage(message);
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
            ws.onclose = () => {
                console.log('WebSocket closed, reconnecting...');
                setTimeout(connectWebSocket, 1000);
            };
        }
        
        function handleMessage(message) {
            if (message.type === 'telemetry') {
                updateMetrics(message.data);
            } else if (message.type === 'state_change') {
                currentState = message.data;
                updateHierarchy(message.data);
                updateLastIntervention(message.data.last_intervention);
            } else if (message.type === 'incident') {
                addIncident(message.data);
            }
        }
        
        function updateHierarchy(state) {
            const tbody = document.getElementById('hierarchy-body');
            tbody.innerHTML = '';
            
            // Org row
            const orgRow = document.createElement('tr');
            orgRow.innerHTML = `<td>Org</td><td>${state.org}</td><td>-</td>`;
            tbody.appendChild(orgRow);
            
            // Service rows
            for (const [serviceName, agents] of Object.entries(state.services)) {
                const serviceRow = document.createElement('tr');
                serviceRow.innerHTML = `<td class="indent-1">Service</td><td>${serviceName}</td><td>-</td>`;
                tbody.appendChild(serviceRow);
                
                // Agent rows
                for (const [agentId, agent] of Object.entries(agents)) {
                    const agentRow = document.createElement('tr');
                    const stateClass = `state-${agent.state}`;
                    agentRow.innerHTML = `
                        <td class="indent-2">Agent</td>
                        <td>${agent.name}</td>
                        <td class="${stateClass}">${agent.state}</td>
                    `;
                    tbody.appendChild(agentRow);
                }
            }
        }
        
        function updateMetrics(telemetry) {
            const tbody = document.getElementById('metrics-body');
            let row = document.getElementById(`metric-${telemetry.agent_id}`);
            
            if (!row) {
                row = document.createElement('tr');
                row.id = `metric-${telemetry.agent_id}`;
                tbody.appendChild(row);
            }
            
            const timestamp = new Date(telemetry.timestamp * 1000).toLocaleTimeString();
            row.innerHTML = `
                <td>${telemetry.agent_id}</td>
                <td>${telemetry.runs_per_min.toFixed(1)}</td>
                <td>${telemetry.tokens_per_run.toFixed(1)}</td>
                <td>${telemetry.confidence_score.toFixed(2)}</td>
                <td>${telemetry.retry_count}</td>
                <td>${timestamp}</td>
            `;
        }
        
        function updateLastIntervention(intervention) {
            const div = document.getElementById('last-intervention');
            if (intervention) {
                div.innerHTML = intervention;
            } else {
                div.innerHTML = '<em>No interventions yet</em>';
            }
        }
        
        function addIncident(incident) {
            const log = document.getElementById('incident-log');
            if (log.innerHTML === '<em>No incidents yet</em>') {
                log.innerHTML = '';
            }
            
            const incidentDiv = document.createElement('div');
            incidentDiv.className = `incident incident-${incident.severity}`;
            const timestamp = new Date(incident.timestamp * 1000).toLocaleString();
            incidentDiv.innerHTML = `
                <strong>${timestamp}</strong> [${incident.agent_id}] ${incident.message}
            `;
            log.insertBefore(incidentDiv, log.firstChild);
        }
        
        async function simulateAnomaly() {
            try {
                const response = await fetch('/simulate/anomaly/classifier', {
                    method: 'POST'
                });
                if (response.ok) {
                    console.log('Anomaly simulation triggered');
                }
            } catch (error) {
                console.error('Error simulating anomaly:', error);
            }
        }
        
        // Initial state fetch
        async function fetchInitialState() {
            try {
                const response = await fetch('/state');
                const state = await response.json();
                currentState = state;
                updateHierarchy(state);
                updateLastIntervention(state.last_intervention);
                
                // Update incidents
                if (state.incidents && state.incidents.length > 0) {
                    const log = document.getElementById('incident-log');
                    log.innerHTML = '';
                    state.incidents.slice().reverse().forEach(incident => {
                        addIncident(incident);
                    });
                }
            } catch (error) {
                console.error('Error fetching initial state:', error);
            }
        }
        
        // Initialize
        fetchInitialState();
        connectWebSocket();
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.get("/state")
async def get_state():
    """Get the complete Helm state."""
    return app_state._serialize_helm_state(app_state.get_helm_state())


@app.post("/simulate/anomaly/{agent_id}")
async def simulate_anomaly(agent_id: str):
    """Simulate an anomaly on a specific agent."""
    worker = app_state.worker_agents.get(agent_id)
    if not worker:
        return {"error": f"Agent {agent_id} not found"}
    
    worker.set_bad_behavior(True)
    agent_info = app_state.agents.get(agent_id)
    if agent_info:
        agent_info.bad_behavior_mode = True
        
    app_state.log_incident(
        agent_id,
        f"Anomaly simulation triggered manually",
        "warning"
    )
    app_state.last_intervention = f"Manual anomaly simulation on {agent_id}"
    app_state.broadcast_state_change()
    
    return {"message": f"Anomaly simulation started for {agent_id}"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    app_state.websocket_connections.add(websocket)
    
    try:
        # Send initial state
        await websocket.send_json({
            "type": "state_change",
            "data": app_state._serialize_helm_state(app_state.get_helm_state())
        })
        
        # Keep connection alive
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        app_state.websocket_connections.discard(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
