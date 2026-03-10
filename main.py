import asyncio
import json
import math
import os
import traceback
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import matplotlib.pyplot as plt
from poliastro.bodies import Earth
from astropy import time
#from poliastro.util import time_range
from poliastro.maneuver import Maneuver
#from poliastro.plotting.plotly import OrbitPlotter3D
#from poliastro.plotting.static import StaticOrbitPlotter
from poliastro.twobody import Orbit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
#from poliastro.plotting.static import StaticOrbitPlotter
from astropy import units as u

# Step 1: Simulate satellite trajectory using Poliastro

import numpy as np
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody.orbit import Orbit

def simulate_trajectory(initial_altitude, initial_velocity):
    """Simulate a satellite trajectory given initial altitude (km) and velocity (km/s).
    Returns orbital parameters (semi-major axis, eccentricity, inclination)."""
    epoch = Time.now()

    r_scalar = Earth.R.to(u.km).value + initial_altitude  # scalar, km
    v_scalar = initial_velocity  # scalar, km/s

    # Build vectors with units
    r = np.array([r_scalar, 0, 0]) * u.km
    v = np.array([0, v_scalar, 0]) * u.km / u.s

    print(f"r: {r}, r.shape: {r.shape}")
    print(f"v: {v}, v.shape: {v.shape}")

    orbit = Orbit.from_vectors(Earth, r, v, epoch)

    # Calculate trajectory for 1000 seconds
    #time_span = time_range(0 * u.s, 1000 * u.s, 100)  # 1000 seconds trajectory
    time_span = [orbit.epoch + i * u.s for i in range(0, 1000, 10)]
    propagated = [orbit.propagate(t - orbit.epoch) for t in time_span]
    #state_vectors = orbit.propagate(time_span)

    # Extract orbital parameters
    semi_major_axis = orbit.a  # semi-major axis in km
    eccentricity = orbit.ecc  # eccentricity
    inclination = orbit.inc  # inclination in radians

    return semi_major_axis.value, eccentricity.value, inclination.value

# Step 2: Generate dataset by simulating multiple trajectories

def generate_dataset(num_samples=100):
    """Generate a dataset by simulating trajectories with random initial conditions."""
    X = []  # list of [altitude, velocity]
    y = []  # list of [semi-major axis, eccentricity, inclination]

    for _ in range(num_samples):
        initial_altitude = np.random.uniform(300, 500)  # Altitude between 300-500 km
        initial_velocity = np.random.uniform(7.0, 8.5)  # Velocity between 7.0-8.5 km/s

        # Simulate the trajectory and get orbital parameters
        semi_major_axis, eccentricity, inclination = simulate_trajectory(initial_altitude, initial_velocity)

        # Append features and labels
        X.append([initial_altitude, initial_velocity])
        y.append([semi_major_axis, eccentricity, inclination])

    # Convert to NumPy arrays **after** data is collected
    X = np.array(X)
    y = np.array(y)
    return X, y


'''def generate_dataset(num_samples=100):
  """Generate a dataset by simulating trajectories with random initial conditions."""
  X = []  # Features (initial altitude, initial velocity)
  y = []  # Labels (orbital parameters: semi-major axis, eccentricity, inclination)

  for _ in range(num_samples):
    # Random initial conditions
    initial_altitude = np.random.uniform(300, 500)  # Altitude between 300-500 km
    initial_velocity = np.random.uniform(7.0, 8.5)  # Velocity between 7.0-8.5 km/s

    # Simulate the trajectory and get orbital parameters
    semi_major_axis, eccentricity, inclination = simulate_trajectory(initial_altitude, initial_velocity)

    # Append to the dataset
    X.append([initial_altitude, initial_velocity])
    y.append([semi_major_axis, eccentricity, inclination])

    X = np.array(X)
    y = np.array(y)

  return X, y'''

# Step 3: Train a machine learning model

def train_model(X, y):
  """ Train a RandomForest model to predict orbital parameters."""
  # Split dataset into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Create and train a RandomForestRegressor model
  model = RandomForestRegressor(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)

  # Predict orbital parameters on the test set
  y_pred = model.predict(X_test)

  # Evaluate the model
  mse = mean_squared_error(y_test, y_pred)
  print(f"Mean Squared Error on Test Set: {mse:.4f}")
  return model, X_test, y_test, y_pred

# Step 4: Visualize the predictions

def plot_results(X_test, y_test, y_pred):
  """ Plot actual vs predicted orbital parameters for comparison."""

  fig, ax = plt.subplots(1, 3, figsize=(18, 5))

  # Plot semi-major axis
  ax[0].scatter(X_test[:, 0], y_test[:, 0], label="True", color="blue", alpha=0.5)
  ax[0].scatter(X_test[:, 0], y_pred[:, 0], label="Predicted", color="red", alpha=0.5)
  ax[0].set_title("Semi-major axis (km)")
  ax[0].set_xlabel("Initial Altitude (km)")
  ax[0].set_ylabel("Semi-major axis (km)")
  ax[0].legend()

  # Plot eccentricity
  ax[1].scatter(X_test[:, 1], y_test[:, 1], label="True", color="blue", alpha=0.5)
  ax[1].scatter(X_test[:, 1], y_pred[:, 1], label="Predicted", color="red", alpha=0.5)
  ax[1].set_title("Eccentricity")
  ax[1].set_xlabel("Initial Velocity (km/s)")
  ax[1].set_ylabel("Eccentricity")
  ax[1].legend()

  # Plot inclination
  ax[2].scatter(X_test[:, 1], y_test[:, 2], label="True", color="blue", alpha=0.5)
  ax[2].scatter(X_test[:, 1], y_pred[:, 2], label="Predicted", color="red", alpha=0.5)
  ax[2].set_title("Inclination (rad)")
  ax[2].set_xlabel("Initial Velocity (km/s)")
  ax[2].set_ylabel("Inclination (rad)")
  ax[2].legend()

  plt.tight_layout()
  plt.show()

# Physics constants (SI units)
G = 6.67430e-11  # gravitational constant, m^3 kg^-1 s^-2
EARTH_MASS = 5.9722e24  # kg
EARTH_RADIUS = 6371000.0  # m
STANDARD_GRAVITY = 9.80665  # m/s^2

# Try to import optional ML model for extended explanations (non-required)
USE_ML_EXPLAINER = False
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    # Optional model name (small/instruction tuned). If you have GPU, it will use it.
    ML_MODEL_NAME = "google/flan-t5-small"  # small instruction model; free
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Torch found. Device: {device}. Attempting to load ML explainer ({ML_MODEL_NAME}) ...")
    tokenizer = AutoTokenizer.from_pretrained(ML_MODEL_NAME)
    ml_model = AutoModelForSeq2SeqLM.from_pretrained(ML_MODEL_NAME).to(device)
    USE_ML_EXPLAINER = True
    print("[Info] ML explainer loaded.")
except Exception as e:
    print("[Info] ML explainer NOT available or failed to load. Continuing without it.")
    # print(traceback.format_exc())
    USE_ML_EXPLAINER = False
    tokenizer = None
    ml_model = None
    device = "cpu"

app = FastAPI(title="Space Calculus Real-time API")

# ------------- Physics / Space utility functions -------------


def gravity_at_radius(radius_m: float, central_mass: float = EARTH_MASS) -> float:
    """
    Compute gravitational acceleration at distance radius_m from center of mass.
    g(r) = G * M / r^2
    """
    return G * central_mass / (radius_m ** 2)


def gravity_at_altitude(altitude_m: float, central_mass: float = EARTH_MASS, central_radius: float = EARTH_RADIUS) -> float:
    """
    Compute gravitational acceleration at altitude above surface.
    altitude_m: meters above surface
    """
    r = central_radius + altitude_m
    return gravity_at_radius(r, central_mass)


def orbital_velocity_circular(radius_m: float, central_mass: float = EARTH_MASS) -> float:
    """
    Circular orbital velocity: v = sqrt(G*M / r)
    """
    return math.sqrt(G * central_mass / radius_m)


def escape_velocity(radius_m: float, central_mass: float = EARTH_MASS) -> float:
    """
    Escape velocity from distance r: v_esc = sqrt(2*G*M / r)
    """
    return math.sqrt(2.0 * G * central_mass / radius_m)


def escape_energy(mass_kg: float, radius_m: float, central_mass: float = EARTH_MASS) -> float:
    """
    Work required to move a mass m from r to infinity (neglecting atmosphere).
    ΔU = G * M * m / r
    """
    return G * central_mass * mass_kg / radius_m


def kinetic_energy_from_velocity(mass_kg: float, velocity_m_s: float) -> float:
    return 0.5 * mass_kg * (velocity_m_s ** 2)


def hohmann_transfer_dv(r1: float, r2: float, mu: float = G * EARTH_MASS) -> Dict[str, float]:
    """
    Hohmann transfer delta-v between two circular orbits at radii r1 and r2.
    Returns dv1 (burn at r1) and dv2 (burn at r2), and total.
    Uses:
      v1 = sqrt(mu / r1)
      v2 = sqrt(mu / r2)
      va = sqrt(mu * (2/r1 - 1/a))  where a = (r1 + r2)/2 (velocity at perigee of transfer ellipse)
      vb = sqrt(mu * (2/r2 - 1/a))  velocity at apogee of transfer ellipse
      dv1 = va - v1
      dv2 = v2 - vb
    """
    a = 0.5 * (r1 + r2)
    v1 = math.sqrt(mu / r1)
    v2 = math.sqrt(mu / r2)
    va = math.sqrt(mu * (2.0 / r1 - 1.0 / a))
    vb = math.sqrt(mu * (2.0 / r2 - 1.0 / a))
    dv1 = abs(va - v1)
    dv2 = abs(v2 - vb)
    return {"dv1_m_s": dv1, "dv2_m_s": dv2, "dv_total_m_s": dv1 + dv2}


def delta_v_budget_estimate(maneuvers: Dict[str, float]) -> float:
    """
    Sum of delta-v per maneuver (input dictionary of maneuvers and their dv in m/s)
    """
    return sum(abs(v) for v in maneuvers.values())


def required_power_for_escape(mass_kg: float, radius_m: float, burn_time_s: float) -> float:
    """
    Rough power estimate to impart the escape energy in a given burn time.
    P = Energy / time
    Energy approximated by escape_energy (work required to move to infinity) or KE of escape
    """
    energy = escape_energy(mass_kg, radius_m)
    if burn_time_s <= 0:
        raise ValueError("burn_time_s must be > 0")
    return energy / burn_time_s


def estimate_launch_cost_per_kg(destination: str = "LEO") -> Dict[str, Any]:
    """
    Heuristic estimates for launch cost per kg to various destinations.
    These are ballpark and depend on provider & launch economics.
    Returns dictionary with ranges in USD/kg.
    (Note: these are heuristics for early-stage design.)
    """
    dest = destination.strip().upper()
    if dest in ("LEO", "LOW EARTH ORBIT"):
        return {"min_usd_per_kg": 1000, "typical_usd_per_kg": 3000, "max_usd_per_kg": 20000}
    elif dest in ("GTO", "GEO TRANSFER", "GEO"):
        return {"min_usd_per_kg": 3000, "typical_usd_per_kg": 8000, "max_usd_per_kg": 40000}
    elif dest in ("LUNAR", "MOON"):
        return {"min_usd_per_kg": 20000, "typical_usd_per_kg": 75000, "max_usd_per_kg": 300000}
    else:
        return {"min_usd_per_kg": 2000, "typical_usd_per_kg": 10000, "max_usd_per_kg": 50000}


def mass_reduction_advice(mass_kg: float) -> Dict[str, Any]:
    """
    Heuristic advice to reduce mass and complexity for a spacecraft.
    Returns recommended percentage reductions for subsystems and textual suggestions.
    """
    # Basic buckets (guidance only)
    advice = [
        {"subsystem": "Structure", "suggest_pct": 10, "notes": "Use composite materials and topology optimization."},
        {"subsystem": "Propulsion", "suggest_pct": 5, "notes": "Optimize tanks, use high Isp engines if feasible."},
        {"subsystem": "Power System", "suggest_pct": 8, "notes": "Right-size solar arrays; integrate batteries where necessary."},
        {"subsystem": "Avionics & Guidance", "suggest_pct": 12, "notes": "Consolidate boards and use low-power modern processors."},
        {"subsystem": "Thermal", "suggest_pct": 6, "notes": "Use passive thermal control, multi-functional panels."},
        {"subsystem": "Payload", "suggest_pct": 7, "notes": "Simplify requirements, use COTS instruments."},
    ]
    # Suggested total possible reduction (heuristic)
    total_possible_pct = sum(x["suggest_pct"] for x in advice)
    total_possible_mass_savings_kg = mass_kg * total_possible_pct / 100.0
    return {"advice": advice, "estimated_total_pct": total_possible_pct, "estimated_mass_savings_kg": total_possible_mass_savings_kg}


# ------------- Input models -------------


class ComputeRequest(BaseModel):
    task: str
    params: dict = {}


# ------------- Helper for nicely formatted numbers -------------


def fmt(number: float, sigfigs: int = 6) -> str:
    """Return a string with scientific notation and given significant digits"""
    if number == 0:
        return "0"
    return f"{number:.{sigfigs}g}"


# ------------- ML-based explanation -------------


async def generate_ml_explanation(prompt: str, max_tokens: int = 256) -> str:
    """
    Optional: uses the small instruction model to generate a richer textual explanation.
    If unavailable, returns the prompt or a short synthesized note.
    """
    if not USE_ML_EXPLAINER:
        return prompt  # fallback: echo prompt or preformatted text
    try:
        input_text = prompt
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        outputs = ml_model.generate(**inputs, max_new_tokens=max_tokens)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text
    except Exception:
        return prompt


# ------------- Endpoints -------------


@app.post("/api/compute")
async def api_compute(req: ComputeRequest):
    #Synchronous HTTP endpoint returning JSON result for a compute task.
    task = req.task.lower()
    p = req.params or {}
    try:
        if task == "gravity_at_altitude":
            alt = float(p.get("altitude_m", 0.0))
            g = gravity_at_altitude(alt)
            return {"task": task, "altitude_m": alt, "g_m_s2": g, "fmt_g": fmt(g)}

        elif task == "orbital_velocity":
            alt = float(p.get("altitude_m", 0.0))
            r = EARTH_RADIUS + alt
            v = orbital_velocity_circular(r)
            period = 2 * math.pi * r / v if v > 0 else None
            return {"task": task, "altitude_m": alt, "radius_m": r, "v_m_s": v, "orbital_period_s": period, "fmt_v": fmt(v)}

        elif task == "escape_velocity":
            alt = float(p.get("altitude_m", 0.0))
            r = EARTH_RADIUS + alt
            v_esc = escape_velocity(r)
            energy_per_kg = escape_energy(1.0, r)  # J per kg
            return {"task": task, "altitude_m": alt, "radius_m": r, "v_escape_m_s": v_esc, "energy_per_kg_J": energy_per_kg, "fmt_v_escape": fmt(v_esc), "fmt_energy_per_kg": fmt(energy_per_kg)}

        elif task == "escape_energy":
            mass = float(p.get("mass_kg", 1.0))
            alt = float(p.get("altitude_m", 0.0))
            r = EARTH_RADIUS + alt
            E = escape_energy(mass, r)
            return {"task": task, "mass_kg": mass, "altitude_m": alt, "energy_J": E, "fmt_energy_J": fmt(E)}

        elif task == "hohmann_transfer":
            alt1 = float(p.get("altitude_from_m", 0.0))
            alt2 = float(p.get("altitude_to_m", 400000.0))
            r1 = EARTH_RADIUS + alt1
            r2 = EARTH_RADIUS + alt2
            dv = hohmann_transfer_dv(r1, r2)
            return {"task": task, "alt_from": alt1, "alt_to": alt2, "dv": dv, "fmt_dv1": fmt(dv["dv1_m_s"]), "fmt_dv2": fmt(dv["dv2_m_s"]), "fmt_dv_total": fmt(dv["dv_total_m_s"])}

        elif task == "power_for_escape":
            mass = float(p.get("mass_kg", 1000.0))
            alt = float(p.get("altitude_m", 0.0))
            burn_time = float(p.get("burn_time_s", 300.0))
            r = EARTH_RADIUS + alt
            P = required_power_for_escape(mass, r, burn_time)
            return {"task": task, "mass_kg": mass, "altitude_m": alt, "burn_time_s": burn_time, "required_power_W": P, "fmt_power_W": fmt(P)}

        elif task == "delta_v_budget":
            maneuvers = p.get("maneuvers", {})
            if not isinstance(maneuvers, dict):
                raise ValueError("maneuvers must be a dict of name->dv_m_s")
            total = delta_v_budget_estimate(maneuvers)
            return {"task": task, "maneuvers": maneuvers, "delta_v_total_m_s": total, "fmt_delta_v_total": fmt(total)}

        elif task == "mass_reduction_advice":
            mass = float(p.get("mass_kg", 100.0))
            advice = mass_reduction_advice(mass)
            return {"task": task, "mass_kg": mass, "advice": advice}

        elif task == "cost_estimate":
            dest = str(p.get("destination", "LEO"))
            mass = float(p.get("mass_kg", 100.0))
            cost_profile = estimate_launch_cost_per_kg(dest)
            est_cost = cost_profile["typical_usd_per_kg"] * mass
            return {"task": task, "destination": dest, "mass_kg": mass, "cost_profile": cost_profile, "estimated_cost_usd": est_cost, "fmt_estimated_cost": fmt(est_cost)}

        else:
            raise ValueError(f"Unknown task: {task}")

    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Computation error: {str(e)}")


# ------------- WebSocket streaming endpoint -------------

@app.get("/")
async def homepage():
    return HTMLResponse(
        "<html><body><h2>Space Calculus API</h2>"
        "<p>Open <a href='/static/index.html'>demo UI</a> to interact.</p></body></html>"
    )

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket message protocol:
      - Client sends JSON: {"task":"...", "params": {...}, "explain": true/false}
      - Server streams JSON messages with:
         {"type":"chunk", "text": "<...>"}  -- partial text
         {"type":"done"} -- finished
         {"type":"result", "json": {...}} -- structured JSON result (sent at end)
         {"type":"error", "message": "..."} -- on error
    """
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            try:
                payload = json.loads(raw)
                task = payload.get("task", "")
                params = payload.get("params", {})
                want_explain = bool(payload.get("explain", True))
            except Exception:
                await ws.send_text(json.dumps({"type": "error", "message": "invalid_json"}))
                continue

            # Compute synchronously but stream a textual explanation progressively
            try:
                # Call compute endpoint logic to get structured result
                compute_req = ComputeRequest(task=task, params=params)
                result = await api_compute(compute_req)
            except Exception as e:
                await ws.send_text(json.dumps({"type": "error", "message": "compute_failed: " + str(e)}))
                await ws.send_text(json.dumps({"type": "done"}))
                continue

            # Prepare an explanation string based on the result
            explanation_lines = []
            explanation_lines.append(f"Task: {task}")
            explanation_lines.append("Inputs:")
            for k, v in params.items():
                explanation_lines.append(f" - {k}: {v}")
            explanation_lines.append("Results (key values):")
            # insert a few human readable numbers from result
            # pick main numeric fields
            numeric_keys = ["g_m_s2", "v_m_s", "v_escape_m_s", "energy_J", "energy_per_kg_J", "required_power_W", "delta_v_total_m_s", "estimated_cost_usd"]
            for nk in numeric_keys:
                if nk in result:
                    explanation_lines.append(f" - {nk}: {result[nk]}")
            # Dump the JSON summary in the explanation
            explanation_lines.append("Full result summary (JSON):")
            explanation_lines.append(json.dumps(result, default=str, indent=2))

            base_explanation = "\n".join(explanation_lines)

            # Optionally run ML-based explainer to add context
            full_explanation = base_explanation
            if want_explain and USE_ML_EXPLAINER:
                # create an instructive prompt
                prompt = f"Explain the following space engineering calculation and give concise design advice:\n\n{base_explanation}\n\nProvide key takeaways and approximations used."
                ml_text = await generate_ml_explanation(prompt, max_tokens=256)
                full_explanation = ml_text

            # Stream the text explanation in small chunks
            chunk_size = 120  # characters
            for i in range(0, len(full_explanation), chunk_size):
                chunk = full_explanation[i : i + chunk_size]
                await ws.send_text(json.dumps({"type": "chunk", "text": chunk}))
                # small sleep to create streaming feel; tune if necessary
                await asyncio.sleep(0.02)

            # After explanation is done, send structured JSON result
            await ws.send_text(json.dumps({"type": "result", "json": result}))
            await ws.send_text(json.dumps({"type": "done"}))

    except WebSocketDisconnect:
        print("WS client disconnected")
    except Exception:
        traceback.print_exc()
        try:
            await ws.send_text(json.dumps({"type": "error", "message": "internal_server_error"}))
            await ws.send_text(json.dumps({"type": "done"}))
        except Exception:
            pass
if __name__ == "__main__":
    import uvicorn
    import nest_asyncio
    from fastapi.responses import RedirectResponse
    from fastapi import Request
    from google.colab.output import eval_js
    import threading

    nest_asyncio.apply()


    # Step 2: Generate dataset
    X, y = generate_dataset(num_samples=500)
    # Step 3: Train the model
    model, X_test, y_test, y_pred = train_model(X, y)
    # Step 4: Visualize results
    plot_results(X_test, y_test, y_pred)
@app.get("/redirect", include_in_schema=False)
    async def root_redirect(request: Request):
        return RedirectResponse(url="/docs")


    def run_uvicorn():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

    threading.Thread(target=run_uvicorn, daemon=True).start()

    print("App running at:", eval_js("google.colab.kernel.proxyPort(8000)"))

    # Prevent the cell from finishing to keep the server alive
    input("Press Enter to stop the server...\n")