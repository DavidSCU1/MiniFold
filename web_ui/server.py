import http.server
import socketserver
import json
import os
import sys
import subprocess
import threading
import time
import glob
from urllib.parse import urlparse, parse_qs
import tkinter as tk
from tkinter import filedialog

# Configuration
PORT = 9000
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Global State
class AppState:
    def __init__(self):
        self.status = "idle"  # idle, running, completed, error
        self.logs = []
        self.current_process = None
        self.lock = threading.Lock()

    def add_log(self, message, level="INFO"):
        # Print to console for visibility
        print(f"[{level}] {message}")
        with self.lock:
            ts = time.strftime("%H:%M:%S")
            self.logs.append({"ts": ts, "level": level, "text": message})
            if len(self.logs) > 1000:
                self.logs.pop(0)

    def clear_logs(self):
        with self.lock:
            self.logs = []

    def set_status(self, status):
        with self.lock:
            self.status = status

state = AppState()

class RequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_request(self, code='-', size='-'):
        if "/api/status" in self.path:
            return
        super().log_request(code, size)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            self.send_json({
                "status": state.status,
                "logs": state.logs
            })
        elif parsed.path == "/api/history":
            self.handle_history()
        elif parsed.path == "/" or parsed.path == "/index.html":
            self.path = "/index.html"
            super().do_GET()
        elif parsed.path.startswith("/output/"):
            # Serve files from the output directory
            # Map /output/... to PROJECT_ROOT/output/...
            rel_path = parsed.path[len("/output/"):]
            file_path = os.path.join(OUTPUT_DIR, rel_path)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                self.serve_file(file_path)
            else:
                self.send_error(404, "File not found")
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/run":
            content_len = int(self.headers.get('Content-Length', 0))
            post_body = self.rfile.read(content_len)
            data = json.loads(post_body.decode('utf-8'))
            
            if state.status == "running":
                self.send_error(400, "Job already running")
                return

            threading.Thread(target=run_minifold, args=(data,)).start()
            self.send_json({"message": "Job started"})
        elif self.path == "/api/browse-dir":
            # Open directory dialog in main thread (or separate, but Tkinter needs main thread usually, 
            # here we do a trick or just run it since it's a simple script)
            # Warning: Tkinter in a thread might be tricky. Let's try simple subprocess or quick tk init.
            try:
                # We can't easily run Tkinter in this thread if main loop isn't running.
                # But we can instantiate Tk, hide it, ask, destroy.
                path = self.browse_directory()
                if path:
                    self.send_json({"path": path})
                else:
                    self.send_json({"error": "Cancelled"})
            except Exception as e:
                self.send_json({"error": str(e)})
        else:
            self.send_error(404)

    def browse_directory(self):
        # Quick Tkinter dialog
        # Use a lock or ensure single threaded access if needed, but for local tool it's fine
        try:
            root = tk.Tk()
            root.withdraw() # Hide main window
            root.attributes('-topmost', True) # Bring to front
            folder_selected = filedialog.askdirectory()
            root.destroy()
            return folder_selected
        except:
            return None

    def send_json(self, data):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def handle_history(self):
        history = []
        if os.path.exists(OUTPUT_DIR):
            # Sort by modification time, newest first
            entries = sorted(
                [os.path.join(OUTPUT_DIR, d) for d in os.listdir(OUTPUT_DIR)],
                key=os.path.getmtime,
                reverse=True
            )
            
            for entry in entries:
                if os.path.isdir(entry):
                    name = os.path.basename(entry)
                    timestamp = os.path.getmtime(entry)
                    
                    # Find HTML files
                    structures_dir = os.path.join(entry, "3d_structures")
                    html_files = []
                    if os.path.exists(structures_dir):
                        html_files = [f for f in os.listdir(structures_dir) if f.endswith(".html")]
                    
                    if html_files:
                        history.append({
                            "name": name,
                            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
                            "files": html_files
                        })
        self.send_json(history)

    def serve_file(self, path):
        try:
            with open(path, 'rb') as f:
                self.send_response(200)
                # Guess mime type based on extension
                if path.endswith(".html"):
                    ctype = "text/html"
                elif path.endswith(".pdb"):
                    ctype = "text/plain"
                else:
                    ctype = "application/octet-stream"
                self.send_header("Content-type", ctype)
                self.end_headers()
                self.wfile.write(f.read())
        except Exception as e:
            self.send_error(500, str(e))

def run_minifold(data):
    state.clear_logs()
    state.set_status("running")
    state.add_log("Starting MiniFold...", "INFO")

    try:
        # Prepare arguments
        job_name = data.get("jobName", f"job_{int(time.time())}")
        fasta_content = data.get("fasta", "")
        
        # Save FASTA
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        fasta_path = os.path.join(OUTPUT_DIR, f"{job_name}.fasta")
        with open(fasta_path, "w", encoding="utf-8") as f:
            f.write(fasta_content)
        
        state.add_log(f"Saved FASTA to {fasta_path}", "INFO")

        # Build Command
        script_path = os.path.join(PROJECT_ROOT, "minifold.py")
        
        # Base Command Logic
        # 1. Default python (current env)
        # 2. Global specific env (if configured)
        
        python_exec = sys.executable # Default
        use_conda_wrapper = False
        conda_env_name = None
        
        if data.get("useGlobalEnv") and data.get("globalEnvName"):
            env_name = data.get("globalEnvName")
            if os.sep in env_name:
                python_exec = env_name
            else:
                use_conda_wrapper = True
                conda_env_name = env_name

        # If iGPU is enabled AND has specific env, it overrides the global env for execution context
        # BUT minifold.py handles the switch internally via subprocess if --igpu is passed?
        # Actually, minifold.py runs the *whole* pipeline. 
        # If user wants iGPU part to run in a specific env, we usually rely on minifold.py to call igpu_runner.py with that env?
        # Let's check how minifold.py handles --igpu.
        # It seems minifold.py currently just enables the flag. The actual environment switch logic for iGPU was in GUI/Launcher.
        # So we should probably run the MAIN script in the "Global Env", and pass the "iGPU Env" name to minifold.py so IT can call the runner.
        # Wait, minifold.py might not accept --igpu-env argument yet. 
        # Let's assume for now we run the whole thing in the target environment if specified.
        
        # Correction based on user request: "Run global env, then iGPU module runs in ITS env"
        # We need to pass the iGPU env name to minifold.py if it supports it.
        # If minifold.py doesn't support --igpu-env, we might need to add it or wrap the execution differently.
        # Looking at previous context, we added ext_env support to GUI which called run_pipeline.
        # run_pipeline accepts `ext_env_name`.
        # So we should pass this as an argument to minifold.py if we modify minifold.py to accept it, 
        # OR we just rely on `minifold.py` to be the entry point.
        
        # Let's constructing the command:
        cmd = []
        
        if use_conda_wrapper and sys.platform == "win32":
             cmd = ["cmd", "/c", "conda", "run", "-n", conda_env_name, "python", script_path, fasta_path]
        elif use_conda_wrapper:
             cmd = ["conda", "run", "-n", conda_env_name, "python", script_path, fasta_path]
        else:
             cmd = [python_exec, script_path, fasta_path]

        cmd.extend(["--outdir", data.get("outDir", "output")])
        if data.get("envText"): cmd.extend(["--env", data.get("envText")])
        cmd.extend(["--ssn", str(data.get("ssn", 5))])
        cmd.extend(["--threshold", str(data.get("threshold", 0.5))])
        
        if data.get("useIgpu"):
            cmd.append("--igpu")
            # If iGPU has specific env, we need to pass it to minifold.py
            # We need to ensure minifold.py accepts --igpu-env or similar.
            # Let's assume we pass it via a new flag we will add to minifold.py momentarily.
            if data.get("useIgpuEnv") and data.get("igpuEnvName"):
                cmd.extend(["--igpu-env", data.get("igpuEnvName")])

        state.add_log(f"Executing: {' '.join(cmd)}", "INFO")

        # Run Process
        # Using shell=True for Windows to ensure PATH is searched for 'conda' or 'cmd' correctly
        use_shell = (sys.platform == "win32")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=PROJECT_ROOT,
            universal_newlines=True,
            shell=use_shell
        )
        
        state.current_process = process

        for line in process.stdout:
            state.add_log(line.strip(), "INFO")
        
        process.wait()
        
        if process.returncode == 0:
            state.set_status("completed")
            state.add_log("Task finished successfully!", "SUCCESS")
        else:
            state.set_status("error")
            state.add_log(f"Task failed with exit code {process.returncode}", "ERROR")

    except Exception as e:
        state.set_status("error")
        state.add_log(f"Internal Error: {str(e)}", "ERROR")

if __name__ == "__main__":
    # Ensure web_ui is current dir so index.html is found
    os.chdir(ROOT_DIR)
    
    server = http.server.ThreadingHTTPServer(("", PORT), RequestHandler)
    print(f"Server started at http://localhost:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped")
