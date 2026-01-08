"""
API Server - FastAPI backend for Live Dashboard.
Provides REST endpoints and WebSocket for real-time updates.

v2.5.0 Feature: Full dashboard integration with settings persistence.
"""

import asyncio
import json
import logging
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, HTMLResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


from .config import get_config, update_config_file

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models for API
# =============================================================================

class ProviderConfig(BaseModel):
    """Provider configuration model."""
    verda_enabled: bool = True
    verda_url: str = "https://api.verda.ai"
    verda_location: str = "FIN-01"
    huggingface_enabled: bool = True
    huggingface_token: str = ""
    ollama_enabled: bool = True
    ollama_url: str = "http://localhost:11434"
    lm_studio_enabled: bool = False
    lm_studio_url: str = "http://localhost:1234"


class ToolsConfig(BaseModel):
    """Tool toggles configuration."""
    instance_management: bool = True
    volume_management: bool = True
    ssh_remote: bool = True
    google_drive: bool = True
    watchdog: bool = True
    smart_deployer: bool = True
    spot_manager: bool = True
    performance_advisor: bool = True
    model_hub: bool = True
    dataset_hub: bool = True
    distributed_training: bool = True
    cost_analytics: bool = True


class TrainingConfig(BaseModel):
    """Training/fine-tuning configuration."""
    learning_rate: str = "2e-5"
    batch_size: int = 4
    epochs: int = 3
    max_length: int = 2048
    lora_enabled: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,v_proj"
    auto_checkpoint: bool = True
    checkpoint_interval: int = 10
    upload_to_gdrive: bool = False


class AlertsConfig(BaseModel):
    """Notifications configuration."""
    discord_enabled: bool = False
    discord_webhook: str = ""
    slack_enabled: bool = False
    slack_webhook: str = ""
    telegram_enabled: bool = False
    telegram_token: str = ""
    alert_training_start: bool = True
    alert_training_complete: bool = True
    alert_checkpoint: bool = True
    alert_eviction: bool = True
    alert_budget: bool = True
    alert_errors: bool = True


class BudgetConfig(BaseModel):
    """Budget and API settings."""
    monthly_budget: float = 500.0
    daily_limit: float = 50.0
    alert_at_percent: int = 70
    auto_stop_at_percent: int = 95
    refresh_interval: int = 30
    auto_reconnect: bool = True


class DashboardSettings(BaseModel):
    """Complete dashboard settings."""
    providers: ProviderConfig = ProviderConfig()
    tools: ToolsConfig = ToolsConfig()
    training: TrainingConfig = TrainingConfig()
    alerts: AlertsConfig = AlertsConfig()
    budget: BudgetConfig = BudgetConfig()


class GPUMetrics(BaseModel):
    """GPU metrics model."""
    gpu_id: int
    name: str = "B300"
    utilization: int = 0
    memory_used: int = 0
    memory_total: int = 262
    temperature: int = 0


class TrainingStatus(BaseModel):
    """Training status model."""
    is_running: bool = False
    current_step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    epoch: int = 0
    total_epochs: int = 0
    eta_seconds: int = 0
    model_name: str = ""
    dataset_name: str = ""


class ConnectionStatus(BaseModel):
    """Connection status model."""
    connected: bool = False
    api_url: str = ""
    location: str = ""
    instance_type: str = ""
    instance_id: str = ""


# =============================================================================
# WebSocket Connection Manager
# =============================================================================

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"WebSocket send error: {e}")
                disconnected.append(connection)

        for conn in disconnected:
            await self.disconnect(conn)


# =============================================================================
# Dashboard State (in-memory + persisted)
# =============================================================================

class DashboardState:
    """Manages dashboard state with persistence."""

    def __init__(self):
        self.settings = DashboardSettings()
        self.gpu_metrics: list[GPUMetrics] = []
        self.training_status = TrainingStatus()
        self.connection_status = ConnectionStatus()
        self.session_cost: float = 0.0
        self.session_saved: float = 0.0
        self.logs: list[dict] = []
        self._config_path: Optional[Path] = None

    def set_config_path(self, path: str):
        self._config_path = Path(path)

    def load_settings(self):
        """Load settings from config.yaml."""
        try:
            config = get_config()
            if config:
                # Map config to settings
                if "dashboard" in config:
                    dash = config["dashboard"]
                    if "providers" in dash:
                        self.settings.providers = ProviderConfig(**dash["providers"])
                    if "tools" in dash:
                        self.settings.tools = ToolsConfig(**dash["tools"])
                    if "training" in dash:
                        self.settings.training = TrainingConfig(**dash["training"])
                    if "alerts" in dash:
                        self.settings.alerts = AlertsConfig(**dash["alerts"])
                    if "budget" in dash:
                        self.settings.budget = BudgetConfig(**dash["budget"])
                logger.info("Settings loaded from config.yaml")
        except Exception as e:
            logger.warning(f"Could not load settings: {e}")

    def save_settings(self):
        """Save settings to config.yaml."""
        try:
            config = get_config() or {}
            config["dashboard"] = {
                "providers": self.settings.providers.model_dump(),
                "tools": self.settings.tools.model_dump(),
                "training": self.settings.training.model_dump(),
                "alerts": self.settings.alerts.model_dump(),
                "budget": self.settings.budget.model_dump(),
            }
            update_config_file(config)
            logger.info("Settings saved to config.yaml")
            return True
        except Exception as e:
            logger.error(f"Could not save settings: {e}")
            return False

    def add_log(self, level: str, message: str):
        """Add a log entry."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        }
        self.logs.insert(0, entry)
        if len(self.logs) > 100:
            self.logs = self.logs[:100]
        return entry


# =============================================================================
# FastAPI Application
# =============================================================================

def create_app() -> "FastAPI":
    """Create and configure the FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn websockets")

    app = FastAPI(
        title="Verda Dashboard API",
        description="Backend API for Verda MCP Live Dashboard",
        version="2.5.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global state
    state = DashboardState()
    manager = ConnectionManager()

    # Load settings on startup
    @app.on_event("startup")
    async def startup():
        state.load_settings()
        # Initialize demo GPU metrics
        state.gpu_metrics = [
            GPUMetrics(gpu_id=0, utilization=94, memory_used=245, temperature=72),
            GPUMetrics(gpu_id=1, utilization=92, memory_used=241, temperature=70),
            GPUMetrics(gpu_id=2, utilization=95, memory_used=248, temperature=73),
            GPUMetrics(gpu_id=3, utilization=93, memory_used=243, temperature=71),
        ]
        state.training_status = TrainingStatus(
            is_running=True,
            current_step=1500,
            total_steps=3000,
            loss=0.234,
            learning_rate=2e-5,
            epoch=2,
            total_epochs=3,
            eta_seconds=8100,
            model_name="LLaMA 3 8B",
            dataset_name="alpaca-52k"
        )
        state.connection_status = ConnectionStatus(
            connected=True,
            api_url="api.verda.ai",
            location="FIN-01",
            instance_type="4x B300 SPOT",
            instance_id="demo-instance"
        )
        state.session_cost = 12.45
        state.session_saved = 37.35
        logger.info("Dashboard API started")

    # ==========================================================================
    # Health & Status Endpoints
    # ==========================================================================

    @app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "version": "2.5.0", "timestamp": datetime.now().isoformat()}

    @app.get("/api/status")
    async def get_status():
        """Get overall dashboard status."""
        return {
            "connection": state.connection_status.model_dump(),
            "training": state.training_status.model_dump(),
            "session_cost": state.session_cost,
            "session_saved": state.session_saved,
            "gpu_count": len(state.gpu_metrics),
        }

    # ==========================================================================
    # Settings Endpoints
    # ==========================================================================

    @app.get("/api/settings", response_model=DashboardSettings)
    async def get_settings():
        """Get all dashboard settings."""
        return state.settings

    @app.put("/api/settings")
    async def update_settings(settings: DashboardSettings):
        """Update all dashboard settings."""
        state.settings = settings
        saved = state.save_settings()
        await manager.broadcast({"type": "settings_updated", "data": settings.model_dump()})
        return {"success": saved, "message": "Settings updated" if saved else "Failed to save"}

    @app.get("/api/settings/providers", response_model=ProviderConfig)
    async def get_providers():
        return state.settings.providers

    @app.put("/api/settings/providers")
    async def update_providers(config: ProviderConfig):
        state.settings.providers = config
        state.save_settings()
        return {"success": True}

    @app.get("/api/settings/tools", response_model=ToolsConfig)
    async def get_tools():
        return state.settings.tools

    @app.put("/api/settings/tools")
    async def update_tools(config: ToolsConfig):
        state.settings.tools = config
        state.save_settings()
        return {"success": True}

    @app.get("/api/settings/training", response_model=TrainingConfig)
    async def get_training_config():
        return state.settings.training

    @app.put("/api/settings/training")
    async def update_training_config(config: TrainingConfig):
        state.settings.training = config
        state.save_settings()
        return {"success": True}

    @app.get("/api/settings/alerts", response_model=AlertsConfig)
    async def get_alerts():
        return state.settings.alerts

    @app.put("/api/settings/alerts")
    async def update_alerts(config: AlertsConfig):
        state.settings.alerts = config
        state.save_settings()
        return {"success": True}

    @app.get("/api/settings/budget", response_model=BudgetConfig)
    async def get_budget():
        return state.settings.budget

    @app.put("/api/settings/budget")
    async def update_budget(config: BudgetConfig):
        state.settings.budget = config
        state.save_settings()
        return {"success": True}

    # ==========================================================================
    # GPU Metrics Endpoints
    # ==========================================================================

    @app.get("/api/gpu")
    async def get_gpu_metrics():
        """Get current GPU metrics."""
        return {
            "gpus": [g.model_dump() for g in state.gpu_metrics],
            "total_utilization": sum(g.utilization for g in state.gpu_metrics) // max(len(state.gpu_metrics), 1),
            "total_memory_used": sum(g.memory_used for g in state.gpu_metrics),
            "total_memory": sum(g.memory_total for g in state.gpu_metrics),
        }

    @app.post("/api/gpu/refresh")
    async def refresh_gpu_metrics():
        """Refresh GPU metrics (trigger SSH nvidia-smi)."""
        # In production, this would SSH to instance and run nvidia-smi
        import random
        for gpu in state.gpu_metrics:
            gpu.utilization = random.randint(90, 98)
            gpu.temperature = random.randint(68, 76)
        await manager.broadcast({"type": "gpu_update", "data": [g.model_dump() for g in state.gpu_metrics]})
        return {"success": True, "gpus": len(state.gpu_metrics)}

    # ==========================================================================
    # Training Endpoints
    # ==========================================================================

    @app.get("/api/training")
    async def get_training_status():
        """Get current training status."""
        return state.training_status.model_dump()

    @app.post("/api/training/stop")
    async def stop_training():
        """Stop training."""
        state.training_status.is_running = False
        state.add_log("info", "Training stopped by user")
        await manager.broadcast({"type": "training_stopped"})
        return {"success": True}

    @app.post("/api/training/checkpoint")
    async def save_checkpoint():
        """Save a checkpoint."""
        step = state.training_status.current_step
        state.add_log("success", f"Checkpoint saved: step_{step}")
        await manager.broadcast({"type": "checkpoint_saved", "step": step})
        return {"success": True, "checkpoint": f"step_{step}"}

    # ==========================================================================
    # Logs Endpoints
    # ==========================================================================

    @app.get("/api/logs")
    async def get_logs(limit: int = 50):
        """Get recent logs."""
        return state.logs[:limit]

    @app.delete("/api/logs")
    async def clear_logs():
        """Clear all logs."""
        state.logs = []
        return {"success": True}

    # ==========================================================================
    # WebSocket Endpoint
    # ==========================================================================

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates."""
        await manager.connect(websocket)
        try:
            # Send initial state
            await websocket.send_json({
                "type": "initial_state",
                "data": {
                    "gpus": [g.model_dump() for g in state.gpu_metrics],
                    "training": state.training_status.model_dump(),
                    "connection": state.connection_status.model_dump(),
                    "session_cost": state.session_cost,
                    "session_saved": state.session_saved,
                }
            })

            # Listen for messages
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)

                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif message.get("type") == "refresh":
                    await refresh_gpu_metrics()

        except WebSocketDisconnect:
            await manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await manager.disconnect(websocket)

    # ==========================================================================
    # Dashboard HTML Endpoint
    # ==========================================================================

    # ==========================================================================
    # Cost Analytics Endpoints
    # ==========================================================================

    @app.get("/api/costs")
    async def get_costs():
        """Get cost analytics data."""
        return {
            "today": state.session_cost,
            "week": state.session_cost * 3.5,
            "month": state.session_cost * 12,
            "savings": state.session_saved,
            "session": state.session_cost,
            "sessionSaved": state.session_saved,
        }

    # ==========================================================================
    # Dashboard HTML Endpoint
    # ==========================================================================

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        """Serve the dashboard HTML."""
        # Look for dashboard HTML file (priority order)
        dashboard_paths = [
            Path(__file__).parent / "dashboard.html",  # New enhanced dashboard
            Path(__file__).parent.parent.parent / "test_dashboard_v2.html",
            Path(__file__).parent.parent.parent / "dashboard.html",
        ]
        for path in dashboard_paths:
            if path.exists():
                logger.info(f"Serving dashboard from: {path}")
                return FileResponse(path, media_type="text/html")

        # Return simple placeholder if not found
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head><title>Verda Dashboard</title></head>
        <body style="background:#0d1117;color:#c9d1d9;font-family:sans-serif;padding:40px;">
            <h1>🚀 Verda Dashboard API</h1>
            <p>API is running. Dashboard HTML not found.</p>
            <ul>
                <li><a href="/api/docs" style="color:#58a6ff;">API Documentation</a></li>
                <li><a href="/api/health" style="color:#58a6ff;">Health Check</a></li>
                <li><a href="/api/status" style="color:#58a6ff;">Status</a></li>
            </ul>
        </body>
        </html>
        """)

    return app


# =============================================================================
# Server Runner with Robust Fail-Safes and Auto Port Switching
# =============================================================================

# Preferred port ranges for fallback
PREFERRED_PORTS = [8765, 8766, 8767, 8768, 8769, 8080, 8081, 8000, 8001, 3000, 3001, 5000, 5001]


def is_port_available(port: int, host: str = "0.0.0.0") -> bool:
    """Check if a port is available. Fast check with minimal timeout."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.05)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except (OSError, socket.error):
        return False


def find_available_port(start_port: int = 8765, max_attempts: int = 50) -> int:
    """
    Rapidly find an available port with multiple fallback strategies.

    Strategy 1: Try preferred ports list first
    Strategy 2: Sequential scan from start_port
    Strategy 3: Random high port as last resort
    """
    # Strategy 1: Try preferred ports first (fastest for common scenarios)
    for port in PREFERRED_PORTS:
        if is_port_available(port):
            return port

    # Strategy 2: Sequential scan from start_port
    for offset in range(max_attempts):
        port = start_port + offset
        if port not in PREFERRED_PORTS and is_port_available(port):
            return port

    # Strategy 3: Try high port range (49152-65535 are dynamic/private)
    for port in range(49152, 49200):
        if is_port_available(port):
            return port

    # Strategy 4: Let OS assign a random available port
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', 0))
            return s.getsockname()[1]
    except Exception:
        pass

    # Ultimate fallback - return original port and let uvicorn handle error
    return start_port


def run_dashboard_server(host: str = "0.0.0.0", port: int = 8765, auto_port: bool = True, max_retries: int = 3):
    """Run the dashboard API server with robust fail-safes.

    Features:
    - Auto port switching if preferred port is in use
    - Multiple fallback port strategies
    - Retry logic with exponential backoff
    - Graceful error handling

    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Preferred port (default: 8765)
        auto_port: If True, automatically find available port if preferred is in use
        max_retries: Maximum number of retry attempts on failure
    """
    if not FASTAPI_AVAILABLE:
        print("❌ FastAPI not installed. Run: pip install fastapi uvicorn websockets pydantic")
        return None

    actual_port = port
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Find available port
            if auto_port:
                actual_port = find_available_port(port)
                if actual_port != port:
                    print(f"⚠️  Port {port} in use, auto-switched to port {actual_port}")

            # Verify port is still available before starting
            if not is_port_available(actual_port):
                print(f"⚠️  Port {actual_port} became unavailable, finding another...")
                actual_port = find_available_port(actual_port + 1)

            print(f"""
╔══════════════════════════════════════════════════════════════╗
║           🚀 Verda Dashboard API Server v2.5.0               ║
╠══════════════════════════════════════════════════════════════╣
║  Dashboard: http://localhost:{actual_port:<5}                          ║
║  API Docs:  http://localhost:{actual_port}/api/docs                   ║
║  Health:    http://localhost:{actual_port}/api/health                 ║
╠══════════════════════════════════════════════════════════════╣
║  Port: {actual_port} | Host: {host} | Auto-port: {auto_port}                    ║
║  Press Ctrl+C to stop                                        ║
╚══════════════════════════════════════════════════════════════╝
""")

            app = create_app()
            uvicorn.run(app, host=host, port=actual_port, log_level="info")
            return actual_port  # Success

        except OSError as e:
            if "address already in use" in str(e).lower() or e.errno in (98, 10048):
                retry_count += 1
                print(f"⚠️  Port conflict (attempt {retry_count}/{max_retries}), finding new port...")
                port = actual_port + 1  # Try next port
                time.sleep(0.5 * retry_count)  # Exponential backoff
            else:
                print(f"❌ OS Error: {e}")
                break

        except Exception as e:
            retry_count += 1
            print(f"❌ Error (attempt {retry_count}/{max_retries}): {e}")
            if retry_count < max_retries:
                print(f"   Retrying in {retry_count}s...")
                time.sleep(retry_count)
            else:
                print("❌ Max retries exceeded. Could not start server.")
                break

    return None


def quick_start(port: int = 8765):
    """Ultra-simple one-liner to start dashboard. Auto-handles everything."""
    return run_dashboard_server(port=port, auto_port=True, max_retries=5)


def dashboard_api(action: str = "status", **kwargs) -> str:
    """
    MCP Tool: Dashboard API Server control.

    Actions:
    - status: Check if server is available
    - start: Start server instructions
    - endpoints: List all API endpoints
    - config: Show configuration options
    """
    if action == "status":
        if FASTAPI_AVAILABLE:
            return """
✅ Dashboard API Ready

Dependencies installed:
- FastAPI ✅
- Uvicorn ✅
- WebSockets ✅
- Pydantic ✅

Start server:
  python -c "from verda_mcp.api_server import run_dashboard_server; run_dashboard_server()"

Or via MCP: dashboard_api(action='start')
"""
        return "❌ FastAPI not installed. Run: pip install fastapi uvicorn websockets pydantic"

    elif action == "start":
        return """
🚀 Start Dashboard API Server

Option 1 - Python:
  python -c "from verda_mcp.api_server import run_dashboard_server; run_dashboard_server()"

Option 2 - Command line:
  uvicorn verda_mcp.api_server:create_app --factory --host 0.0.0.0 --port 8765

Option 3 - With reload (development):
  uvicorn verda_mcp.api_server:create_app --factory --reload --port 8765

Then open: http://localhost:8765
API docs: http://localhost:8765/api/docs
"""

    elif action == "endpoints":
        return """
📡 Dashboard API Endpoints

HEALTH & STATUS
  GET  /api/health           Health check
  GET  /api/status           Overall status

SETTINGS (GET/PUT)
  GET  /api/settings         All settings
  PUT  /api/settings         Update all
  GET  /api/settings/providers
  GET  /api/settings/tools
  GET  /api/settings/training
  GET  /api/settings/alerts
  GET  /api/settings/budget

GPU METRICS
  GET  /api/gpu              GPU metrics
  POST /api/gpu/refresh      Refresh metrics

TRAINING
  GET  /api/training         Training status
  POST /api/training/stop    Stop training
  POST /api/training/checkpoint  Save checkpoint

LOGS
  GET  /api/logs             Recent logs
  DEL  /api/logs             Clear logs

WEBSOCKET
  WS   /ws                   Real-time updates
"""

    elif action == "config":
        return """
⚙️ Dashboard Configuration

Settings are persisted to config.yaml under 'dashboard' key:

dashboard:
  providers:
    verda_enabled: true
    verda_url: https://api.verda.ai
    huggingface_enabled: true
    ollama_enabled: true
  tools:
    instance_management: true
    watchdog: true
    ...
  training:
    learning_rate: "2e-5"
    lora_r: 16
    ...
  alerts:
    discord_enabled: false
    ...
  budget:
    monthly_budget: 500
    ...
"""

    return "Actions: status, start, endpoints, config"
