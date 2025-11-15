import os
import sys
import subprocess
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = "secret"
socketio = SocketIO(app, cors_allowed_origins="*")

simulation_process = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)

@socketio.on("start_simulation")
def start_simulation():
    global simulation_process
    
    if simulation_process is None or simulation_process.poll() is not None:
        try:
            script_path = os.path.join(os.getcwd(), "drone_sim_mujoco.py")

            simulation_process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            def forward_output():
                for line in simulation_process.stdout:
                    socketio.emit("simulation_output", {"data": line.strip()})

            threading.Thread(target=forward_output, daemon=True).start()
            
            emit("simulation_status", {"status": "started"})
        except Exception as e:
            emit("simulation_status", {"status": "error", "message": str(e)})
    else:
        emit("simulation_status", {"status": "already_running"})

@socketio.on("stop_simulation")
def stop_simulation():
    global simulation_process
    if simulation_process and simulation_process.poll() is None:
        simulation_process.terminate()
        simulation_process = None
        emit("simulation_status", {"status": "stopped"})

if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000)
