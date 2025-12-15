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
import importlib.util

# Configuration
PORT = 9000
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Global State
class AppState:
    def __init__(self):
        self.status = "idle"  # idle, running, completed, error
        self.stop_requested = False
        self.logs = []
        self.current_process = None
        self.progress = 0
        self.current_step = ""
        self.start_time = None
        self.lock = threading.Lock()
        self.output_dir_rel = "output"
        self.output_dir_abs = OUTPUT_DIR

    def add_log(self, message, level="INFO"):
        # Print to console for visibility
        print(f"[{level}] {message}")
        with self.lock:
            ts = time.strftime("%H:%M:%S")
            self.logs.append({"ts": ts, "level": level, "text": message})
            if len(self.logs) > 1000:
                self.logs.pop(0)

    def update_progress(self, percent, step):
        with self.lock:
            self.progress = percent
            self.current_step = step

    def clear_logs(self):
        with self.lock:
            self.logs = []
            self.progress = 0
            self.current_step = ""
            self.start_time = None

    def set_status(self, status):
        with self.lock:
            self.status = status
            if status == "running":
                self.stop_requested = False
                self.start_time = time.time()
                self.progress = 0
                self.current_step = "Starting..."

state = AppState()

class RequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_request(self, code='-', size='-'):
        if "/api/status" in self.path:
            return
        super().log_request(code, size)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            self.handle_status()
        elif parsed.path == "/api/trajectory":
            base_dir = None
            latest_time = 0
            latest_pdb = None
            if os.path.exists(state.output_dir_abs):
                for root, dirs, files in os.walk(state.output_dir_abs):
                    for file in files:
                        if file.endswith(".pdb") or file.endswith(".pdb.tmp"):
                            full_path = os.path.join(root, file)
                            try:
                                mtime = os.path.getmtime(full_path)
                                if mtime > latest_time:
                                    latest_time = mtime
                                    latest_pdb = full_path
                               
                            except:
                                pass
            content_parts = []
            if latest_pdb:
                base_dir = os.path.dirname(latest_pdb)
                frames_dir = os.path.join(base_dir, "_frames")
                if os.path.exists(frames_dir):
                    try:
                        frame_files = [f for f in os.listdir(frames_dir) if f.endswith(".pdb")]
                        frame_files.sort()
                        idx = 1
                        for fn in frame_files:
                            fp = os.path.join(frames_dir, fn)
                            text = None
                            for _ in range(5):
                                try:
                                    with open(fp, "r", encoding="utf-8") as f:
                                        text = f.read()
                                    break
                                except PermissionError:
                                    time.sleep(0.02)
                                except Exception:
                                    break
                            if text:
                                content_parts.append(f"MODEL {idx}\n")
                                content_parts.append(text)
                                if not text.endswith("\n"):
                                    content_parts.append("\n")
                                content_parts.append("ENDMDL\n")
                                idx += 1
                    except:
                        pass
            body = "".join(content_parts) if content_parts else ""
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
        elif parsed.path == "/api/history":
            self.handle_history()
        elif parsed.path == "/api/files":
            self.handle_file_list()
        elif parsed.path == "/" or parsed.path == "/index.html":
            self.serve_file(os.path.join(ROOT_DIR, "index.html"))
        elif parsed.path.startswith("/output/"):
            # Serve files from the output directory
            # Map /output/... to PROJECT_ROOT/output/...
            # Fix: The original logic was parsed.path[len("/output/"):] which removes "output/"
            # But OUTPUT_DIR is PROJECT_ROOT/output
            # So if URL is /output/job/file.pdb, rel_path is job/file.pdb
            # And file_path becomes PROJECT_ROOT/output/job/file.pdb
            # This seems correct.
            
            # However, let's double check path construction
            rel_path = parsed.path[len("/output/"):]
            
            # Decode URL components (e.g. %20 -> space)
            from urllib.parse import unquote
            rel_path = unquote(rel_path)
            
            file_path = os.path.join(state.output_dir_abs, rel_path)
            
            # print(f"DEBUG: Serving {parsed.path} -> {file_path}")
            # print(f"DEBUG: Serving {parsed.path} -> {file_path}")
            
            if os.path.exists(file_path) and os.path.isfile(file_path):
                self.serve_file(file_path)
            else:
                # print(f"DEBUG: File not found: {file_path}")
                # print(f"DEBUG: File not found: {file_path}")
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

            out_dir_val = (data.get("outDir", "output") or "output").strip()
            if os.path.isabs(out_dir_val):
                if os.path.abspath(out_dir_val).lower().startswith(PROJECT_ROOT.lower()):
                    out_rel = os.path.relpath(os.path.abspath(out_dir_val), PROJECT_ROOT)
                else:
                    out_rel = os.path.join("output", os.path.basename(out_dir_val))
            else:
                out_rel = out_dir_val
            abs_path = os.path.abspath(os.path.join(PROJECT_ROOT, out_rel))
            try:
                os.makedirs(abs_path, exist_ok=True)
            except Exception:
                out_rel = "output"
                abs_path = OUTPUT_DIR
                os.makedirs(abs_path, exist_ok=True)
            with state.lock:
                state.output_dir_rel = out_rel.replace("\\", "/")
                state.output_dir_abs = abs_path

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
        elif self.path == "/api/read-file":
            content_len = int(self.headers.get('Content-Length', 0))
            post_body = self.rfile.read(content_len)
            data = json.loads(post_body.decode('utf-8'))
            self.handle_read_file(data)
        elif self.path == "/api/stop":
            if state.status == "running" and state.current_process:
                state.stop_requested = True
                try:
                    state.add_log("Stopping job...", "WARNING")
                    if sys.platform == "win32":
                        subprocess.call(['taskkill', '/F', '/T', '/PID', str(state.current_process.pid)])
                    else:
                        state.current_process.terminate()
                    self.send_json({"status": "stopped", "message": "Job stopped"})
                except Exception as e:
                    self.send_error(500, str(e))
            else:
                self.send_error(400, "No running job to stop")
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

    def handle_status(self):
        # Scan for latest PDB in OUTPUT_DIR to update preview
        # This is a bit hacky but works for live preview without modifying backend too much
        # Find the most recently modified .pdb file in output dir recursively
        
        # Compute elapsed time for progress panel
        elapsed = 0
        try:
            if state.status == "running" and state.start_time:
                elapsed = time.time() - state.start_time
        except:
            elapsed = 0
        
        latest_pdb = None
        latest_pdb_mtime = 0
        latest_rel = None
        latest_preview_url = None
        trajectory_frames_count = 0
        trajectory_last_mtime = 0
        try:
            if os.path.exists(state.output_dir_abs):
                latest_time = 0
                for root, dirs, files in os.walk(state.output_dir_abs):
                    for file in files:
                        if file.endswith(".pdb") or file.endswith(".pdb.tmp"):
                            full_path = os.path.join(root, file)
                            try:
                                mtime = os.path.getmtime(full_path)
                                # Force float comparison and ensure we pick up any file if none selected
                                if mtime > latest_time:
                                    latest_time = mtime
                                    latest_pdb = full_path
                                    latest_pdb_mtime = mtime
                                    latest_rel = os.path.relpath(full_path, state.output_dir_abs).replace("\\", "/")
                                    latest_preview_url = f"/output/{latest_rel}"
                            except:
                                pass
                if latest_pdb:
                    base_dir = os.path.dirname(latest_pdb)
                    frames_dir = os.path.join(base_dir, "_frames")
                    if os.path.exists(frames_dir):
                        try:
                            frame_files = [f for f in os.listdir(frames_dir) if f.endswith(".pdb")]
                            trajectory_frames_count = len(frame_files)
                            if frame_files:
                                max_m = 0
                                for fn in frame_files:
                                    fp = os.path.join(frames_dir, fn)
                                    try:
                                        m = os.path.getmtime(fp)
                                        if m > max_m:
                                            max_m = m
                                    except:
                                        pass
                                trajectory_last_mtime = max_m
                                latest_pdb_mtime = max(latest_pdb_mtime, max_m)
                                latest_preview_url = "/api/trajectory"
                        except:
                            pass
                
                # Debug print if running
                if state.status == "running" and latest_pdb:
                     # print(f"DEBUG: Found latest PDB: {latest_pdb} ({latest_pdb_mtime})")
                     # print(f"DEBUG: Found latest PDB: {latest_pdb} ({latest_pdb_mtime})")
                     pass
                elif state.status == "running" and not latest_pdb:
                     # print("DEBUG: No PDB found yet")
                     # print("DEBUG: No PDB found yet")
                     pass
                     
        except:
            pass

        self.send_json({
            "status": state.status,
            "logs": state.logs,
            "progress": state.progress,
            "current_step": state.current_step,
            "elapsed_time": elapsed,
            "latest_pdb": latest_pdb,
            "latest_pdb_mtime": latest_pdb_mtime,
            "latest_rel": latest_rel,
            "latest_preview_url": latest_preview_url,
            "trajectory_frames_count": trajectory_frames_count,
            "trajectory_last_mtime": trajectory_last_mtime,
            "output_dir_rel": state.output_dir_rel
        })
        
    def handle_read_file(self, data):
        path = data.get("path")
        if not path:
            self.send_error(404, "File not found")
            return
            
        # If path is absolute, use it directly (after security check)
        # If path is relative, assume it's relative to OUTPUT_DIR? 
        # But handle_status sends absolute paths.
        
        # Security check: must be inside PROJECT_ROOT
        # Normalize for Windows (case insensitive)
        abs_path = os.path.abspath(path).lower()
        abs_root = PROJECT_ROOT.lower()
        
        # print(f"DEBUG: Reading file {abs_path}")
        # print(f"DEBUG: Reading file {abs_path}")
        
        if not abs_path.startswith(abs_root):
             print(f"Access denied: {abs_path} not in {abs_root}")
             self.send_error(403, "Access denied")
             return
             
        if not os.path.exists(path):
             self.send_error(404, "File not found")
             return

        # Retry logic for Windows file locking
        content = None
        for _ in range(5):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                break
            except PermissionError:
                time.sleep(0.05)
            except Exception:
                pass
        
        if content is None:
            self.send_error(503, "File locked or inaccessible")
            return

        try:
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content.encode("utf-8"))
        except Exception as e:
            self.send_error(500, str(e))

    def handle_history(self):
        history = []
        base_dir = state.output_dir_abs
        if os.path.exists(base_dir):
            # Sort by modification time, newest first
            entries = sorted(
                [os.path.join(base_dir, d) for d in os.listdir(base_dir)],
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

    def handle_file_list(self):
        """
        Recursively list files in output directory for the file explorer.
        Returns a tree structure:
        [
            {
                "name": "folder1",
                "type": "dir",
                "children": [...]
            },
            {
                "name": "file.txt",
                "type": "file"
            }
        ]
        """
        def build_tree(path):
            tree = []
            try:
                # Sort: Directories first, then files
                entries = os.listdir(path)
                entries.sort(key=lambda x: (not os.path.isdir(os.path.join(path, x)), x.lower()))
                
                for entry in entries:
                    full_path = os.path.join(path, entry)
                    rel_path = os.path.relpath(full_path, state.output_dir_abs).replace("\\", "/")
                    
                    item = {
                        "name": entry,
                        "path": rel_path
                    }
                    
                    if os.path.isdir(full_path):
                        item["type"] = "dir"
                        item["children"] = build_tree(full_path)
                    else:
                        item["type"] = "file"
                        
                    tree.append(item)
            except Exception as e:
                print(f"Error listing dir {path}: {e}")
            return tree

        base_dir = state.output_dir_abs
        if os.path.exists(base_dir):
            tree = build_tree(base_dir)
        else:
            tree = []
        self.send_json(tree)

    def serve_file(self, path):
        # Retry logic for Windows file locking
        content = None
        for _ in range(5):
            try:
                with open(path, 'rb') as f:
                    content = f.read()
                break
            except PermissionError:
                time.sleep(0.05)
            except Exception as e:
                self.send_error(500, str(e))
                return

        if content is None:
            self.send_error(503, "File locked or inaccessible")
            return

        try:
            self.send_response(200)
            # Guess mime type based on extension
            if path.endswith(".html"):
                ctype = "text/html"
            elif path.endswith(".pdb"):
                ctype = "text/plain"
            elif path.endswith(".js"):
                ctype = "application/javascript"
            elif path.endswith(".css"):
                ctype = "text/css"
            elif path.endswith(".json"):
                ctype = "application/json"
            elif path.endswith(".txt"):
                ctype = "text/plain"
            elif path.endswith(".tmp"):
                ctype = "text/plain"
            else:
                ctype = "application/octet-stream"
            self.send_header("Content-type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
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
        if not os.path.exists(state.output_dir_abs):
            os.makedirs(state.output_dir_abs)
        
        fasta_path = os.path.join(state.output_dir_abs, f"{job_name}.fasta")
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
        # cmd = [] # Moved below
        
        # NOTE: We construct the command cleanly here to avoid logic duplication above
        if sys.platform == "win32":
            if use_conda_wrapper:
                # "cmd /c conda run --no-capture-output -n env python -u script ..."
                cmd = ["cmd", "/c", "conda", "run", "--no-capture-output", "-n", conda_env_name, "python", "-u", script_path, fasta_path]
                
                # Append common args
                cmd.extend(["--outdir", state.output_dir_abs])
                if data.get("envText"):
                    cmd.extend(["--env", data.get("envText")])
                cmd.extend(["--ssn", str(data.get("ssn", 5))])
                cmd.extend(["--threshold", str(data.get("threshold", 0.5))])
                
                if data.get("useIgpu"):
                    cmd.append("--igpu")
                    if data.get("useIgpuEnv") and data.get("igpuEnvName"):
                        cmd.extend(["--igpu-env", data.get("igpuEnvName")])
                
                # Better to pass as string for shell=True
                cmd_str = subprocess.list2cmdline(cmd)
                use_shell = True
                cmd_to_run = cmd_str
            else:
                # Direct python call
                cmd = [python_exec, "-u", script_path, fasta_path]
                cmd.extend(["--outdir", state.output_dir_abs])
                if data.get("envText"):
                    cmd.extend(["--env", data.get("envText")])
                cmd.extend(["--ssn", str(data.get("ssn", 5))])
                cmd.extend(["--threshold", str(data.get("threshold", 0.5))])
                
                if data.get("useIgpu"):
                    cmd.append("--igpu")
                    if data.get("useIgpuEnv") and data.get("igpuEnvName"):
                        cmd.extend(["--igpu-env", data.get("igpuEnvName")])

                cmd_to_run = cmd
                use_shell = False
        else:
             # Linux/Mac
             if use_conda_wrapper:
                 cmd = ["conda", "run", "--no-capture-output", "-n", conda_env_name, "python", "-u", script_path, fasta_path]
             else:
                 cmd = [python_exec, "-u", script_path, fasta_path]
             
             cmd.extend(["--outdir", state.output_dir_abs])
             if data.get("envText"): cmd.extend(["--env", data.get("envText")])
             cmd.extend(["--ssn", str(data.get("ssn", 5))])
             cmd.extend(["--threshold", str(data.get("threshold", 0.5))])
             
             if data.get("useIgpu"):
                cmd.append("--igpu")
                if data.get("useIgpuEnv") and data.get("igpuEnvName"):
                    cmd.extend(["--igpu-env", data.get("igpuEnvName")])

             cmd_to_run = cmd
             use_shell = False

        state.add_log(f"Executing: {cmd_to_run}", "INFO")

        process = subprocess.Popen(
            cmd_to_run,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=PROJECT_ROOT,
            universal_newlines=True,
            shell=use_shell,
            # Ensure we don't inherit handles that might conflict
            close_fds=(sys.platform != "win32") 
        )
        
        state.current_process = process
        
        # Read output
        import re
        progress_pattern = re.compile(r"\[PROGRESS\]\s+(\d+)%\s+-\s+(.*)")
        
        for line in iter(process.stdout.readline, ''):
            if line:
                stripped = line.strip()
                match = progress_pattern.search(stripped)
                if match:
                    percent = int(match.group(1))
                    step = match.group(2)
                    state.update_progress(percent, step)
                    state.add_log(f"Progress: {percent}% - {step}", "INFO")
                else:
                    # Filter spam logs if needed
                    state.add_log(stripped, "INFO")
                    # Fallback check for debug:
                    if "[PROGRESS]" in stripped:
                         state.add_log(f"DEBUG: Found [PROGRESS] but regex failed on: '{stripped}'", "WARNING")
        
        process.wait()
        
        if state.stop_requested:
            state.set_status("stopped")
            state.add_log("Job stopped by user.", "WARNING")
        elif process.returncode == 0:
            state.set_status("completed")
            state.add_log("Job completed successfully.", "SUCCESS")
        else:
            state.set_status("error")
            state.add_log(f"Job failed with exit code {process.returncode}", "ERROR")

    except Exception as e:
        state.set_status("error")
        state.add_log(f"Server Error: {str(e)}", "ERROR")
        import traceback
        state.add_log(traceback.format_exc(), "ERROR")
        

if __name__ == "__main__":
    # Ensure output dir exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    try:
        server_cls = http.server.ThreadingHTTPServer
    except Exception:
        server_cls = http.server.HTTPServer
    server = server_cls(('0.0.0.0', PORT), RequestHandler)
    print(f"Server started at http://localhost:{PORT}")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()
