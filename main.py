"""
Helm - Hierarchical Control Plane for AI Agents
A single-process FastAPI application with REAL autonomous agents.
"""
import asyncio
import json
import os
import queue
import random
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set
from uuid import uuid4

from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# ============================================================================
# Configuration & API Clients
# ============================================================================

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is required")

claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Tonic API configuration (using a simple data generator if Tonic API not available)
TONIC_API_KEY = os.getenv("TONIC_API_KEY", "")
TONIC_API_URL = os.getenv("TONIC_API_URL", "https://api.tonic.ai/v1")


# ============================================================================
# Data Models
# ============================================================================

class AgentState(str, Enum):
    HEALTHY = "healthy"
    PAUSED = "paused"
    DEGRADED = "degraded"


@dataclass
class Telemetry:
    """Agent telemetry data from real execution."""
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
    rate_limit: float = 1.0  # tasks per second
    max_retries: int = 3
    batch_size: int = 1


@dataclass
class HelmState:
    """Complete Helm hierarchy and state."""
    org: str
    services: Dict[str, Dict[str, AgentInfo]]
    incidents: List[Incident]
    last_intervention: Optional[str] = None


# ============================================================================
# Tonic Data Generator (Synthetic Data)
# ============================================================================

class TonicDataGenerator:
    """Generates realistic synthetic data for agent processing."""
    
    @staticmethod
    def generate_customer_record() -> dict:
        """Generate a synthetic customer record for classification."""
        risk_factors = ["low", "medium", "high"]
        return {
            "customer_id": f"CUST-{random.randint(10000, 99999)}",
            "age": random.randint(18, 80),
            "income": random.randint(20000, 200000),
            "credit_score": random.randint(300, 850),
            "account_age_months": random.randint(0, 120),
            "transaction_count": random.randint(0, 1000),
            "risk_category": random.choice(risk_factors),
            "notes": f"Customer with {random.choice(['stable', 'volatile', 'growing'])} transaction history"
        }
    
    @staticmethod
    def generate_incident_report() -> str:
        """Generate a synthetic incident report for summarization."""
        incidents = [
            "System outage occurred at 14:30 UTC affecting payment processing. Root cause: database connection pool exhaustion. Impact: 15% of transactions failed. Resolution: Restarted connection pool, restored service at 15:45 UTC.",
            "Security alert triggered for suspicious login attempts from IP 192.168.1.100. Multiple failed authentication attempts detected. Action taken: IP blocked, user account locked pending review.",
            "Performance degradation in API response times. Average latency increased from 200ms to 1200ms. Investigation revealed memory leak in caching layer. Fix deployed, monitoring continues.",
            "Data synchronization failure between primary and replica databases. Last successful sync: 2 hours ago. Automatic retry initiated, manual intervention may be required.",
            "Third-party service dependency (payment gateway) experiencing intermittent failures. Error rate: 8%. Fallback mechanism activated, primary service monitoring continues."
        ]
        return random.choice(incidents)
    
    @staticmethod
    def generate_malformed_data() -> dict:
        """Generate malformed data to trigger errors."""
        return {
            "invalid": "data",
            "missing_fields": True,
            "corrupted": None
        }


# ============================================================================
# Real Worker Agents
# ============================================================================

class ClassifierAgent:
    """Real agent that classifies customer records using Claude."""
    
    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self.bad_behavior_mode = False
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.telemetry_queue: deque = deque(maxlen=300)  # Keep last 5 minutes
        self.execution_times: deque = deque(maxlen=60)  # For runs_per_min calculation
        self.rate_limit = 1.0
        self.max_retries = 3
        self.batch_size = 1
        
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
            self.thread.join(timeout=5.0)
    
    def set_bad_behavior(self, enabled: bool):
        """Enable bad behavior mode (increases batch size, introduces errors)."""
        self.bad_behavior_mode = enabled
        if enabled:
            self.batch_size = 10  # Large batch to increase token usage
            self.rate_limit = 5.0  # Faster rate
    
    def _run(self):
        """Main agent loop - performs real classification work."""
        while self.running:
            try:
                # Check if paused
                agent_info = app_state.agents.get(self.agent_id)
                if agent_info and agent_info.state == AgentState.PAUSED:
                    time.sleep(1.0)
                    continue
                
                # Execute classification task
                start_time = time.time()
                telemetry = self._classify_records()
                execution_time = time.time() - start_time
                
                if telemetry:
                    self.execution_times.append(execution_time)
                    self.telemetry_queue.append(telemetry)
                    
                    # Update global state
                    if agent_info:
                        agent_info.current_telemetry = telemetry
                        app_state.broadcast_telemetry(telemetry)
                
                # Rate limiting
                time.sleep(1.0 / self.rate_limit)
                
            except Exception as e:
                print(f"Error in ClassifierAgent: {e}")
                time.sleep(1.0)
    
    def _classify_records(self) -> Optional[Telemetry]:
        """Classify customer records using Claude. Returns telemetry."""
        retry_count = 0
        tokens_used = 0
        confidence = 0.0
        
        # Generate data (malformed if bad behavior mode)
        if self.bad_behavior_mode and random.random() < 0.3:
            # 30% chance of malformed data
            data = TonicDataGenerator.generate_malformed_data()
            data_str = json.dumps(data)
        else:
            records = [TonicDataGenerator.generate_customer_record() for _ in range(self.batch_size)]
            data_str = json.dumps(records, indent=2)
        
        prompt = f"""Classify the following customer record(s) into risk categories (low, medium, high).

Customer data:
{data_str}

Provide:
1. Risk category for each customer
2. Brief reasoning
3. Confidence score (0.0 to 1.0)

Format as JSON with keys: classifications, reasoning, confidence"""
        
        max_attempts = self.max_retries + 1
        for attempt in range(max_attempts):
            try:
                response = claude_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Extract token usage
                tokens_used = response.usage.input_tokens + response.usage.output_tokens
                
                # Parse response for confidence
                content = response.content[0].text
                try:
                    # Try to extract confidence from response
                    if "confidence" in content.lower():
                        # Simple extraction
                        import re
                        conf_match = re.search(r'confidence["\']?\s*:\s*([0-9.]+)', content, re.IGNORECASE)
                        if conf_match:
                            confidence = float(conf_match.group(1))
                        else:
                            confidence = 0.8  # Default if parsing fails
                    else:
                        confidence = 0.8
                except:
                    confidence = 0.8
                
                # In bad behavior mode, artificially lower confidence
                if self.bad_behavior_mode:
                    confidence *= 0.6
                
                break  # Success
                
            except Exception as e:
                retry_count += 1
                if retry_count >= max_attempts:
                    # Failed after retries
                    tokens_used = 100  # Estimate
                    confidence = 0.3
                    break
                time.sleep(0.5)
        
        # Calculate runs_per_min from execution times
        if len(self.execution_times) > 0:
            avg_execution_time = statistics.mean(self.execution_times)
            runs_per_min = 60.0 / avg_execution_time if avg_execution_time > 0 else 0
        else:
            runs_per_min = 0
        
        return Telemetry(
            agent_id=self.agent_id,
            timestamp=time.time(),
            runs_per_min=runs_per_min,
            tokens_per_run=tokens_used / max(1, self.batch_size),
            confidence_score=confidence,
            retry_count=retry_count
        )
    
    def get_recent_telemetry(self, seconds: int = 10) -> List[Telemetry]:
        """Get telemetry from the last N seconds."""
        cutoff = time.time() - seconds
        return [t for t in self.telemetry_queue if t.timestamp >= cutoff]


class ProcessorAgent:
    """Real agent that processes and summarizes incident reports using Claude."""
    
    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self.bad_behavior_mode = False
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.telemetry_queue: deque = deque(maxlen=300)
        self.execution_times: deque = deque(maxlen=60)
        self.rate_limit = 0.8
        self.max_retries = 3
        self.batch_size = 1
        
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
            self.thread.join(timeout=5.0)
    
    def set_bad_behavior(self, enabled: bool):
        """Enable bad behavior mode."""
        self.bad_behavior_mode = enabled
        if enabled:
            self.batch_size = 5
            self.rate_limit = 3.0
    
    def _run(self):
        """Main agent loop - performs real summarization work."""
        while self.running:
            try:
                agent_info = app_state.agents.get(self.agent_id)
                if agent_info and agent_info.state == AgentState.PAUSED:
                    time.sleep(1.0)
                    continue
                
                start_time = time.time()
                telemetry = self._process_reports()
                execution_time = time.time() - start_time
                
                if telemetry:
                    self.execution_times.append(execution_time)
                    self.telemetry_queue.append(telemetry)
                    
                    if agent_info:
                        agent_info.current_telemetry = telemetry
                        app_state.broadcast_telemetry(telemetry)
                
                time.sleep(1.0 / self.rate_limit)
                
            except Exception as e:
                print(f"Error in ProcessorAgent: {e}")
                time.sleep(1.0)
    
    def _process_reports(self) -> Optional[Telemetry]:
        """Process incident reports using Claude. Returns telemetry."""
        retry_count = 0
        tokens_used = 0
        confidence = 0.0
        
        # Generate reports
        reports = [TonicDataGenerator.generate_incident_report() for _ in range(self.batch_size)]
        reports_text = "\n\n---\n\n".join(reports)
        
        prompt = f"""Summarize the following incident report(s). Provide:
1. Key issues identified
2. Impact assessment
3. Resolution status
4. Confidence score (0.0 to 1.0) in your summary

Incident reports:
{reports_text}

Format as JSON with keys: summary, impact, resolution, confidence"""
        
        max_attempts = self.max_retries + 1
        for attempt in range(max_attempts):
            try:
                response = claude_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                tokens_used = response.usage.input_tokens + response.usage.output_tokens
                
                content = response.content[0].text
                try:
                    import re
                    conf_match = re.search(r'confidence["\']?\s*:\s*([0-9.]+)', content, re.IGNORECASE)
                    if conf_match:
                        confidence = float(conf_match.group(1))
                    else:
                        confidence = 0.85
                except:
                    confidence = 0.85
                
                if self.bad_behavior_mode:
                    confidence *= 0.5
                
                break
                
            except Exception as e:
                retry_count += 1
                if retry_count >= max_attempts:
                    tokens_used = 150
                    confidence = 0.4
                    break
                time.sleep(0.5)
        
        if len(self.execution_times) > 0:
            avg_execution_time = statistics.mean(self.execution_times)
            runs_per_min = 60.0 / avg_execution_time if avg_execution_time > 0 else 0
        else:
            runs_per_min = 0
        
        return Telemetry(
            agent_id=self.agent_id,
            timestamp=time.time(),
            runs_per_min=runs_per_min,
            tokens_per_run=tokens_used / max(1, self.batch_size),
            confidence_score=confidence,
            retry_count=retry_count
        )
    
    def get_recent_telemetry(self, seconds: int = 10) -> List[Telemetry]:
        """Get telemetry from the last N seconds."""
        cutoff = time.time() - seconds
        return [t for t in self.telemetry_queue if t.timestamp >= cutoff]


# ============================================================================
# Observer Agent (Real Analysis)
# ============================================================================

class ObserverAgent:
    """Real observer that analyzes telemetry with rolling baselines."""
    
    def __init__(self):
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.baselines: Dict[str, deque] = {}  # Rolling baselines per agent
        
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
        """Main observer loop - analyzes real telemetry."""
        while self.running:
            try:
                for agent_id, agent_info in app_state.agents.items():
                    if agent_info.state == AgentState.PAUSED:
                        continue
                    
                    worker = app_state.worker_agents.get(agent_id)
                    if not worker:
                        continue
                    
                    recent_telemetry = worker.get_recent_telemetry(seconds=30)
                    if len(recent_telemetry) < 3:
                        continue
                    
                    # Update rolling baseline
                    if agent_id not in self.baselines:
                        self.baselines[agent_id] = deque(maxlen=100)
                    
                    # Add recent token usage to baseline
                    for tel in recent_telemetry:
                        self.baselines[agent_id].append(tel.tokens_per_run)
                    
                    # Detect drift using rolling baseline
                    drift_detected = self._detect_drift(agent_id, agent_info, recent_telemetry)
                    if drift_detected:
                        app_state.handle_drift(agent_id, drift_detected)
                
                time.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                print(f"Error in ObserverAgent: {e}")
                time.sleep(5.0)
    
    def _detect_drift(self, agent_id: str, agent_info: AgentInfo, telemetry: List[Telemetry]) -> Optional[str]:
        """Detect drift using rolling baselines. Returns drift reason or None."""
        if not telemetry:
            return None
        
        latest = telemetry[-1]
        baseline_queue = self.baselines.get(agent_id, deque())
        
        if len(baseline_queue) < 10:
            # Not enough data for baseline
            return None
        
        # Calculate rolling baseline (median of recent values)
        baseline_values = list(baseline_queue)[-20:]  # Last 20 values
        baseline_median = statistics.median(baseline_values) if baseline_values else agent_info.baseline_tokens
        
        # Condition 1: Token usage > 5× rolling baseline
        if baseline_median > 0 and latest.tokens_per_run > baseline_median * 5.0:
            return f"Token usage {latest.tokens_per_run:.1f} exceeds 5× rolling baseline {baseline_median:.1f}"
        
        # Condition 2: Confidence < 0.55 for N consecutive runs
        low_confidence_count = sum(1 for t in telemetry[-5:] if t.confidence_score < 0.55)
        if low_confidence_count >= 3:
            return f"Low confidence detected: {low_confidence_count} consecutive runs below 0.55"
        
        # Condition 3: Repeated retries
        retry_count = sum(1 for t in telemetry[-5:] if t.retry_count > 0)
        if retry_count >= 3:
            return f"Repeated retries detected: {retry_count} runs with retries"
        
        return None


# ============================================================================
# Helm Governors (Autonomous Agents)
# ============================================================================

class AgentHelm:
    """Autonomous agent-level Helm governor."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start the Helm governor as an autonomous agent."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the Helm governor."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def _run(self):
        """Autonomous monitoring and intervention loop."""
        while self.running:
            try:
                agent_info = app_state.agents.get(self.agent_id)
                if not agent_info:
                    time.sleep(2.0)
                    continue
                
                # If agent is paused, attempt recovery
                if agent_info.state == AgentState.PAUSED:
                    self._attempt_recovery()
                
                time.sleep(3.0)  # Check every 3 seconds
                
            except Exception as e:
                print(f"Error in AgentHelm {self.agent_id}: {e}")
                time.sleep(3.0)
    
    def handle_drift(self, reason: str) -> bool:
        """Handle drift detection. Returns True if recovered."""
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
        
        return self._attempt_recovery()
    
    def _attempt_recovery(self) -> bool:
        """Attempt to recover the agent. Returns True if successful."""
        agent_info = app_state.agents.get(self.agent_id)
        if not agent_info:
            return False
        
        worker = app_state.worker_agents.get(self.agent_id)
        if not worker:
            return False
        
        self.recovery_attempts += 1
        
        if self.recovery_attempts <= self.max_recovery_attempts:
            # Modify operating constraints
            worker.set_bad_behavior(False)
            worker.batch_size = 1
            worker.rate_limit = 1.0
            worker.max_retries = 3
            
            time.sleep(5.0)  # Wait for recovery
            
            # Check if recovered
            recent = worker.get_recent_telemetry(seconds=10)
            if recent and len(recent) > 0:
                latest = recent[-1]
                baseline_queue = app_state.observer.baselines.get(self.agent_id, deque())
                if len(baseline_queue) >= 10:
                    baseline = statistics.median(list(baseline_queue)[-20:])
                else:
                    baseline = agent_info.baseline_tokens
                
                if (latest.tokens_per_run <= baseline * 5.0 and
                    latest.confidence_score >= 0.55 and
                    latest.retry_count <= 2):
                    agent_info.state = AgentState.HEALTHY
                    self.recovery_attempts = 0
                    app_state.log_incident(
                        self.agent_id,
                        f"Agent recovered successfully (attempt {self.recovery_attempts})",
                        "info"
                    )
                    app_state.last_intervention = f"Agent {agent_info.name} recovered by AgentHelm"
                    app_state.broadcast_state_change()
                    return True
        
        # Recovery failed - escalate
        agent_info.state = AgentState.DEGRADED
        app_state.log_incident(
            self.agent_id,
            f"Agent recovery failed after {self.recovery_attempts} attempts. Escalating to ServiceHelm.",
            "error"
        )
        app_state.last_intervention = f"Agent {agent_info.name} escalated to ServiceHelm"
        app_state.broadcast_state_change()
        return False


class ServiceHelm:
    """Autonomous service-level Helm governor."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.agent_helms: Dict[str, AgentHelm] = {}
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start the service-level Helm as an autonomous agent."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the service-level Helm."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def _run(self):
        """Autonomous service-level monitoring."""
        while self.running:
            try:
                # Monitor for escalations and service-wide issues
                degraded_count = sum(
                    1 for agent_info in app_state.agents.values()
                    if agent_info.state == AgentState.DEGRADED
                )
                
                if degraded_count >= 2:
                    # Multiple agents degraded - service-level intervention
                    app_state.log_incident(
                        "service",
                        f"Service-wide degradation detected: {degraded_count} agents degraded",
                        "critical"
                    )
                    app_state.last_intervention = f"ServiceHelm: Service-wide intervention for {self.service_name}"
                    app_state.broadcast_state_change()
                
                time.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                print(f"Error in ServiceHelm: {e}")
                time.sleep(10.0)
    
    def register_agent(self, agent_id: str):
        """Register an agent under this service."""
        self.agent_helms[agent_id] = AgentHelm(agent_id)
    
    def handle_escalation(self, agent_id: str):
        """Handle escalation from an agent-level Helm."""
        agent_info = app_state.agents.get(agent_id)
        if not agent_info:
            return
        
        app_state.log_incident(
            agent_id,
            f"ServiceHelm intervention for {agent_info.name} in service {self.service_name}",
            "critical"
        )
        app_state.last_intervention = f"ServiceHelm intervention for {agent_info.name}"
        app_state.broadcast_state_change()
        
        # Service-level recovery: force reset
        worker = app_state.worker_agents.get(agent_id)
        if worker:
            worker.set_bad_behavior(False)
            worker.batch_size = 1
            worker.rate_limit = 0.5  # Slow down
            worker.max_retries = 2
            
            time.sleep(5.0)
            
            recent = worker.get_recent_telemetry(seconds=10)
            if recent:
                latest = recent[-1]
                baseline_queue = app_state.observer.baselines.get(agent_id, deque())
                if len(baseline_queue) >= 10:
                    baseline = statistics.median(list(baseline_queue)[-20:])
                else:
                    baseline = agent_info.baseline_tokens
                
                if (latest.tokens_per_run <= baseline * 5.0 and
                    latest.confidence_score >= 0.55 and
                    latest.retry_count <= 1):
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
        self.worker_agents: Dict[str, any] = {}  # ClassifierAgent or ProcessorAgent
        self.agent_helms: Dict[str, AgentHelm] = {}
        self.service_helms: Dict[str, ServiceHelm] = {}
        self.incidents: List[Incident] = []
        self.last_intervention: Optional[str] = None
        self.websocket_connections: Set[WebSocket] = set()
        self.observer: Optional[ObserverAgent] = None
        self.message_queue: queue.Queue = queue.Queue()
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        
    def initialize(self):
        """Initialize the application with real agents."""
        # Create service
        service_name = "ai-service"
        self.services[service_name] = {}
        
        # Create service-level Helm
        service_helm = ServiceHelm(service_name)
        self.service_helms[service_name] = service_helm
        service_helm.start()
        
        # Create real agents
        classifier = ClassifierAgent("classifier", "Classifier Agent")
        processor = ProcessorAgent("processor", "Processor Agent")
        
        agents_config = [
            ("classifier", "Classifier Agent", classifier, 200.0),
            ("processor", "Processor Agent", processor, 300.0),
        ]
        
        for agent_id, name, worker, baseline_tokens in agents_config:
            agent_info = AgentInfo(
                agent_id=agent_id,
                name=name,
                state=AgentState.HEALTHY,
                baseline_tokens=baseline_tokens
            )
            self.agents[agent_id] = agent_info
            self.services[service_name][agent_id] = agent_info
            
            # Store worker agent
            self.worker_agents[agent_id] = worker
            worker.start()
            
            # Create and start agent-level Helm
            agent_helm = AgentHelm(agent_id)
            self.agent_helms[agent_id] = agent_helm
            agent_helm.start()
            service_helm.register_agent(agent_id)
            
        # Start observer
        self.observer = ObserverAgent()
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
        """Serialize HelmState to dict."""
        result = {
            "org": state.org,
            "services": {},
            "incidents": [asdict(inc) for inc in state.incidents],
            "last_intervention": state.last_intervention
        }
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

app = FastAPI(title="Helm - Real Agent Runtime Control Plane")


async def broadcast_worker():
    """Background task to process WebSocket broadcasts."""
    while True:
        try:
            try:
                message = app_state.message_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.1)
                continue
                
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
        print("Helm application started successfully with REAL agents")
    except Exception as e:
        print(f"Error during startup: {e}")
        import traceback
        traceback.print_exc()


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    for worker in app_state.worker_agents.values():
        worker.stop()
    for helm in app_state.agent_helms.values():
        helm.stop()
    for helm in app_state.service_helms.values():
        helm.stop()
    if app_state.observer:
        app_state.observer.stop()


@app.get("/")
async def get_ui():
    """Serve the control plane UI with real-time charts."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Helm - Agent Runtime Control Plane</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
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
        .chart-container {
            background: #2a2a2a;
            padding: 15px;
            margin: 20px 0;
            border: 1px solid #444;
            height: 300px;
        }
    </style>
</head>
<body>
    <h1>Helm — Agent Runtime Control Plane</h1>
    
    <button onclick="simulateAnomaly()">Simulate anomaly on classifier</button>
    
    <h2>Talk to Agents</h2>
    <div style="background: #2a2a2a; padding: 15px; border: 1px solid #444; margin: 20px 0;">
        <div style="margin-bottom: 10px;">
            <label style="display: block; margin-bottom: 5px;">Select Agent:</label>
            <select id="agent-select" style="background: #1a1a1a; color: #e0e0e0; border: 1px solid #444; padding: 5px; width: 200px;">
                <option value="classifier">Classifier Agent</option>
                <option value="processor">Processor Agent</option>
            </select>
        </div>
        <div style="margin-bottom: 10px;">
            <label style="display: block; margin-bottom: 5px;">Your Message:</label>
            <textarea id="chat-input" style="width: 100%; background: #1a1a1a; color: #e0e0e0; border: 1px solid #444; padding: 10px; min-height: 60px; font-family: monospace;" placeholder="Ask the agent a question..."></textarea>
        </div>
        <button onclick="sendMessage()" style="background: #6ac259; margin-top: 10px;">Send Message</button>
        <div id="chat-response" style="margin-top: 15px; padding: 10px; background: #1a1a1a; border: 1px solid #444; min-height: 50px; display: none;">
            <strong style="color: #4a9eff;">Agent Response:</strong>
            <div id="chat-response-text" style="margin-top: 10px; white-space: pre-wrap;"></div>
            <div id="chat-response-meta" style="margin-top: 10px; font-size: 12px; color: #888;"></div>
        </div>
    </div>
    
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
    
    <h2>Token Usage Over Time</h2>
    <div class="chart-container">
        <canvas id="tokensChart"></canvas>
    </div>
    
    <h2>Confidence Over Time</h2>
    <div class="chart-container">
        <canvas id="confidenceChart"></canvas>
    </div>
    
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
        
        // Chart data storage
        const chartData = {
            classifier: { tokens: [], confidence: [], timestamps: [] },
            processor: { tokens: [], confidence: [], timestamps: [] }
        };
        const maxDataPoints = 50;
        
        // Initialize charts
        const tokensChart = new Chart(document.getElementById('tokensChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Classifier Tokens',
                        data: [],
                        borderColor: '#4a9eff',
                        backgroundColor: 'rgba(74, 158, 255, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Processor Tokens',
                        data: [],
                        borderColor: '#6ac259',
                        backgroundColor: 'rgba(106, 194, 89, 0.1)',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#e0e0e0' } }
                },
                scales: {
                    x: { ticks: { color: '#e0e0e0' }, grid: { color: '#444' } },
                    y: { ticks: { color: '#e0e0e0' }, grid: { color: '#444' } }
                }
            }
        });
        
        const confidenceChart = new Chart(document.getElementById('confidenceChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Classifier Confidence',
                        data: [],
                        borderColor: '#4a9eff',
                        backgroundColor: 'rgba(74, 158, 255, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Processor Confidence',
                        data: [],
                        borderColor: '#6ac259',
                        backgroundColor: 'rgba(106, 194, 89, 0.1)',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#e0e0e0' } }
                },
                scales: {
                    x: { ticks: { color: '#e0e0e0' }, grid: { color: '#444' } },
                    y: { 
                        ticks: { color: '#e0e0e0' }, 
                        grid: { color: '#444' },
                        min: 0,
                        max: 1
                    }
                }
            }
        });
        
        function updateCharts(telemetry) {
            const agentId = telemetry.agent_id;
            const time = new Date(telemetry.timestamp * 1000).toLocaleTimeString();
            
            if (!chartData[agentId]) return;
            
            // Add data
            chartData[agentId].tokens.push(telemetry.tokens_per_run);
            chartData[agentId].confidence.push(telemetry.confidence_score);
            chartData[agentId].timestamps.push(time);
            
            // Limit data points
            if (chartData[agentId].tokens.length > maxDataPoints) {
                chartData[agentId].tokens.shift();
                chartData[agentId].confidence.shift();
                chartData[agentId].timestamps.shift();
            }
            
            // Update tokens chart
            if (agentId === 'classifier') {
                tokensChart.data.labels = chartData.classifier.timestamps;
                tokensChart.data.datasets[0].data = chartData.classifier.tokens;
            } else if (agentId === 'processor') {
                tokensChart.data.datasets[1].data = chartData.processor.tokens;
                if (tokensChart.data.labels.length === 0) {
                    tokensChart.data.labels = chartData.processor.timestamps;
                }
            }
            tokensChart.update('none');
            
            // Update confidence chart
            if (agentId === 'classifier') {
                confidenceChart.data.labels = chartData.classifier.timestamps;
                confidenceChart.data.datasets[0].data = chartData.classifier.confidence;
            } else if (agentId === 'processor') {
                confidenceChart.data.datasets[1].data = chartData.processor.confidence;
                if (confidenceChart.data.labels.length === 0) {
                    confidenceChart.data.labels = chartData.processor.timestamps;
                }
            }
            confidenceChart.update('none');
        }
        
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
                updateCharts(message.data);
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
            
            const orgRow = document.createElement('tr');
            orgRow.innerHTML = `<td>Org</td><td>${state.org}</td><td>-</td>`;
            tbody.appendChild(orgRow);
            
            for (const [serviceName, agents] of Object.entries(state.services)) {
                const serviceRow = document.createElement('tr');
                serviceRow.innerHTML = `<td class="indent-1">Service</td><td>${serviceName}</td><td>-</td>`;
                tbody.appendChild(serviceRow);
                
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
        
        async function sendMessage() {
            const agentId = document.getElementById('agent-select').value;
            const query = document.getElementById('chat-input').value.trim();
            
            if (!query) {
                alert('Please enter a message');
                return;
            }
            
            const responseDiv = document.getElementById('chat-response');
            const responseText = document.getElementById('chat-response-text');
            const responseMeta = document.getElementById('chat-response-meta');
            
            responseDiv.style.display = 'block';
            responseText.textContent = 'Sending...';
            responseMeta.textContent = '';
            
            try {
                const response = await fetch(`/agent/${agentId}/query`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ query: query })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    responseText.textContent = `Error: ${data.error}`;
                    responseText.style.color = '#ff4444';
                } else {
                    responseText.textContent = data.response;
                    responseText.style.color = '#e0e0e0';
                    responseMeta.textContent = `Tokens used: ${data.tokens_used} | Agent: ${data.agent_id}`;
                    responseMeta.style.color = '#888';
                    
                    // Clear input
                    document.getElementById('chat-input').value = '';
                }
            } catch (error) {
                responseText.textContent = `Error: ${error.message}`;
                responseText.style.color = '#ff4444';
            }
        }
        
        // Allow Enter key to send
        document.addEventListener('DOMContentLoaded', function() {
            const chatInput = document.getElementById('chat-input');
            if (chatInput) {
                chatInput.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' && e.ctrlKey) {
                        sendMessage();
                    }
                });
            }
        });
        
        async function fetchInitialState() {
            try {
                const response = await fetch('/state');
                const state = await response.json();
                currentState = state;
                updateHierarchy(state);
                updateLastIntervention(state.last_intervention);
                
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
    """Simulate an anomaly on a specific agent - causes REAL failures."""
    worker = app_state.worker_agents.get(agent_id)
    if not worker:
        return {"error": f"Agent {agent_id} not found"}
    
    # Enable bad behavior mode - causes real issues
    worker.set_bad_behavior(True)
    agent_info = app_state.agents.get(agent_id)
    if agent_info:
        agent_info.bad_behavior_mode = True
        
    app_state.log_incident(
        agent_id,
        f"Real anomaly injection triggered - increased batch size, malformed data enabled",
        "warning"
    )
    app_state.last_intervention = f"Manual anomaly injection on {agent_id}"
    app_state.broadcast_state_change()
    
    return {"message": f"Real anomaly injection started for {agent_id}"}


class QueryRequest(BaseModel):
    query: str


@app.post("/agent/{agent_id}/query")
async def query_agent(agent_id: str, request: QueryRequest):
    """Send a query to an agent and get a response using Claude."""
    query = request.query
    if not query:
        return {"error": "Query is required"}
    
    agent_info = app_state.agents.get(agent_id)
    if not agent_info:
        return {"error": f"Agent {agent_id} not found"}
    
    # Build agent-specific prompt
    if agent_id == "classifier":
        system_prompt = "You are a Classifier Agent specialized in analyzing customer records and classifying them into risk categories (low, medium, high). You help users understand risk assessment and classification logic."
        user_prompt = f"User query: {query}\n\nPlease provide a helpful response based on your classification expertise."
    elif agent_id == "processor":
        system_prompt = "You are a Processor Agent specialized in summarizing and analyzing incident reports. You help users understand incidents, their impact, and resolution status."
        user_prompt = f"User query: {query}\n\nPlease provide a helpful response based on your incident processing expertise."
    else:
        system_prompt = "You are an AI agent assistant."
        user_prompt = f"User query: {query}\n\nPlease provide a helpful response."
    
    try:
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        
        answer = response.content[0].text
        tokens_used = response.usage.input_tokens + response.usage.output_tokens
        
        # Log the interaction
        app_state.log_incident(
            agent_id,
            f"User query processed: {query[:50]}... (tokens: {tokens_used})",
            "info"
        )
        
        return {
            "agent_id": agent_id,
            "query": query,
            "response": answer,
            "tokens_used": tokens_used,
            "timestamp": time.time()
        }
    except Exception as e:
        return {"error": f"Failed to process query: {str(e)}"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    app_state.websocket_connections.add(websocket)
    
    try:
        await websocket.send_json({
            "type": "state_change",
            "data": app_state._serialize_helm_state(app_state.get_helm_state())
        })
        
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        app_state.websocket_connections.discard(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
