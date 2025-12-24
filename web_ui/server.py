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
import sys
if 'sys' in globals():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(ROOT_DIR)
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)

# Configuration
PORT = 9000
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import types

def resolve_conda_env_to_python(env_name):
    """
    Attempt to resolve a conda environment name to its python executable path.
    Returns None if not found.
    """
    # 1. Check if env_name is already a path
    if os.sep in env_name or "/" in env_name:
        if os.path.exists(env_name):
             # check for python.exe
             if os.path.isdir(env_name):
                 py = os.path.join(env_name, "python.exe")
                 if os.path.exists(py): return py
                 py = os.path.join(env_name, "bin", "python") # linux/mac
                 if os.path.exists(py): return py
             elif os.path.isfile(env_name) and "python" in os.path.basename(env_name):
                 return env_name
        return None

    # 2. Use conda info --json
    try:
        # Use shell=True for windows to find conda
        cmd = ["conda", "info", "--envs", "--json"]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', shell=True if sys.platform=='win32' else False)
        if res.returncode != 0:
            return None
        
        data = json.loads(res.stdout)
        envs = data.get("envs", [])
        
        # Search for name match
        for env_path in envs:
            if os.path.basename(env_path).lower() == env_name.lower():
                # Found it
                py = os.path.join(env_path, "python.exe")
                if os.path.exists(py): return py
                py = os.path.join(env_path, "bin", "python")
                if os.path.exists(py): return py
        
        # Handle "base"
        if env_name.lower() == "base":
             root = data.get("root_prefix")
             if root:
                 py = os.path.join(root, "python.exe")
                 if os.path.exists(py): return py
                 py = os.path.join(root, "bin", "python")
                 if os.path.exists(py): return py
    except:
        pass
    return None

def ensure_local_modules_package():
    modules_dir = os.path.join(PROJECT_ROOT, "modules")
    m = sys.modules.get("modules")
    if m is None or not getattr(m, "__path__", None):
        pkg = types.ModuleType("modules")
        pkg.__path__ = [modules_dir]
        sys.modules["modules"] = pkg
    else:
        p = list(getattr(m, "__path__", []))
        if modules_dir not in p:
            m.__path__ = [modules_dir] + p

# Global State
class AppState:
    def __init__(self):
        self.status = "idle"  # idle, running, completed, error
        self.stop_requested = False
        self.logs = []
        self.current_process = None
        self.progress = 0
        self.current_step = ""
        self.detail_progress = ""  # New field for granular progress
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

    def update_progress(self, percent, step, detail=None):
        with self.lock:
            self.progress = percent
            self.current_step = step
            if detail is not None:
                self.detail_progress = detail

    def clear_logs(self):
        with self.lock:
            self.logs = []
            self.progress = 0
            self.current_step = ""
            self.detail_progress = ""
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
                            max_idx = 0
                            for fn in frame_files:
                                try:
                                    name = fn.split(".")[0]
                                    if name.startswith("frame_"):
                                        num = int(name[6:])
                                        if num > max_idx:
                                            max_idx = num
                                except:
                                    pass
                            trajectory_frames_count = max_idx
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
            "detail_progress": state.detail_progress,
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
        try:
            from modules.env_loader import load_env
            load_env(os.path.join(PROJECT_ROOT, ".env"))
        except Exception:
            pass
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

        use_global_env = bool(data.get("useGlobalEnv") and data.get("globalEnvName"))
        global_env_name = data.get("globalEnvName")
        use_direct_3d = bool(data.get("direct3D"))
        direct_ss_dir = data.get("ssDir")

        def logger(msg: str):
            text = str(msg).strip()
            if not text: return
            
            # PARSE PROGRESS TAGS FROM MINIFOLD STDOUT
            # Format: [PROGRESS] 30% - Step Name
            if "[PROGRESS]" in text:
                try:
                    # Extract percentage and step name
                    # Example: "[PROGRESS] 30% - Generated SS candidates"
                    parts = text.split("[PROGRESS]")[-1].strip().split("%")
                    if len(parts) >= 2:
                        pct = int(parts[0].strip())
                        step_name = parts[1].strip(" -:")
                        state.update_progress(pct, step_name)
                except:
                    pass
            
            # PARSE DETAILED PROGRESS (Optimization Steps)
            # Format: [PROGRESS_DETAIL] Phase 1 (Coarse): Step 5/50 | Loss: 204.3
            elif "[PROGRESS_DETAIL]" in text:
                try:
                    detail = text.split("[PROGRESS_DETAIL]")[-1].strip()
                    state.update_progress(state.progress, state.current_step, detail=detail)
                except:
                    pass

            # Fallback heuristic for other logs (optional, but [PROGRESS] tag is preferred)
            elif "读取 FASTA" in text or "Loaded FASTA" in text:
                state.update_progress(5, "Loaded FASTA sequences")
            elif "候选生成完成" in text:
                state.update_progress(30, "Generated SS candidates")
            elif "Ark 投票完成" in text:
                state.update_progress(45, "Voting and selection")
            elif "功能注释完成" in text:
                state.update_progress(60, "Sequence annotation")
            elif "生成 3D 结构" in text:
                state.update_progress(65, "Generating 3D structures")
            elif "报告已生成" in text:
                state.update_progress(95, "Report generated")
                
            state.add_log(text, "INFO")

        if use_global_env:
            if use_direct_3d:
                state.add_log("Direct 3D mode currently仅支持本地内置环境运行，已忽略全局环境设置。", "WARNING")
            state.add_log(f"Running in external environment: {global_env_name}", "INFO")
            
            # Construct command for minifold.py
            cmd_args = [
                "minifold.py",
                fasta_path,
                "--outdir", state.output_dir_abs,
                "--ssn", str(data.get("ssn", 5)),
                "--threshold", str(data.get("threshold", 0.5))
            ]
            
            if data.get("envText"):
                cmd_args.extend(["--env", data.get("envText")])
                
            if bool(data.get("useIgpu")):
                cmd_args.append("--igpu")
                # Add backend argument if present
                backend = data.get("backend", "auto")
                if backend and backend != "auto":
                    cmd_args.extend(["--backend", backend])
                    
                # If running minifold.py externally, we pass igpu-env to it, 
                # and it will handle the iGPU runner subprocess.
                if bool(data.get("useIgpuEnv") and data.get("igpuEnvName")):
                    cmd_args.extend(["--igpu-env", data.get("igpuEnvName")])
            
            if bool(data.get("useNpu")):
                cmd_args.append("--npu")
                if bool(data.get("useNpuEnv") and data.get("npuEnvName")):
                    cmd_args.extend(["--npu-env", data.get("npuEnvName")])
            
            if bool(data.get("useEsmBackbone")):
                cmd_args.append("--esm-backbone")
                if bool(data.get("useEsmEnv") and data.get("esmEnvName")):
                    cmd_args.extend(["--esm-env", data.get("esmEnvName")])
            
            # Pass target chains if present
            target_chains_val = data.get("targetChains")
            try:
                # Convert to int to check validity, but keep as string for cmd
                tc_int = int(target_chains_val)
                if tc_int > 0:
                    cmd_args.extend(["--target-chains", str(tc_int)])
                    state.add_log(f"Enforcing Target Chains: {tc_int}", "INFO")
            except:
                pass

            # Conda wrapper logic
            final_cmd = []
            
            # Try to resolve python path first for better stability and unbuffered output
            python_path = resolve_conda_env_to_python(global_env_name)
            
            if python_path:
                state.add_log(f"Resolved environment '{global_env_name}' to: {python_path}", "INFO")
                # Use -u for unbuffered output
                final_cmd = [python_path, "-u"] + cmd_args
            elif os.sep in global_env_name or "/" in global_env_name:
                # Path to python executable provided directly
                final_cmd = [global_env_name, "-u"] + cmd_args
            else:
                # Fallback to conda run
                # We use 'conda run -n <env> python -u ...'
                # Note: On Windows, capturing stdout from 'conda run' can sometimes be buffered or encoding-sensitive.
                final_cmd = ["conda", "run", "-n", global_env_name, "python", "-u"] + cmd_args

            # Execute with subprocess
            try:
                # On Windows, using shell=True for conda might be necessary if conda is not in PATH directly but via shell hook
                use_shell = (sys.platform == "win32")
                
                # Setup environment variables to force UTF-8
                env_vars = os.environ.copy()
                env_vars["PYTHONIOENCODING"] = "utf-8"
                env_vars["PYTHONUTF8"] = "1"
                
                process = subprocess.Popen(
                    final_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=PROJECT_ROOT,
                    bufsize=1,
                    encoding='utf-8',
                    errors='replace',
                    shell=use_shell,
                    env=env_vars
                )
                state.current_process = process
                
                # Read output loop
                while True:
                    if state.stop_requested:
                        process.terminate()
                        break
                        
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        logger(line.strip())
                
                rc = process.poll()
                if rc == 0:
                    state.set_status("completed")
                    state.update_progress(100, "Completed")
                    state.add_log("Job completed successfully.", "SUCCESS")
                else:
                    state.set_status("error")
                    state.add_log(f"Job failed with exit code {rc}", "ERROR")
            except Exception as e:
                state.set_status("error")
                state.add_log(f"Subprocess execution failed: {str(e)}", "ERROR")

        else:
            # Execute pipeline in-process (Original Logic)
            try:
                ensure_local_modules_package()
                use_ext_env = bool(data.get("useIgpuEnv") and data.get("igpuEnvName"))
                use_igpu = bool(data.get("useIgpu"))
                use_npu = bool(data.get("useNpu"))
                use_npu_ext_env = bool(data.get("useNpuEnv") and data.get("npuEnvName"))
                use_esm = bool(data.get("useEsmBackbone"))
                use_esm_ext_env = bool(use_esm and data.get("useEsmEnv") and data.get("esmEnvName"))
                import modules.pipeline
                import modules.input_handler
                importlib.reload(modules.input_handler)
                if use_igpu and not use_ext_env:
                    import modules.igpu_predictor
                    importlib.reload(modules.igpu_predictor)
                importlib.reload(modules.pipeline)
                
                from modules.pipeline import run_pipeline
                from modules.env_loader import load_env
                load_env(os.path.join(PROJECT_ROOT, ".env"))
                target_chains_val = data.get("targetChains")
                try:
                    if target_chains_val:
                        target_chains_val = int(target_chains_val)
                    else:
                        target_chains_val = None
                except:
                    target_chains_val = None
    
                run_pipeline(
                    fasta_path,
                    state.output_dir_abs,
                    data.get("envText"),
                    int(data.get("ssn", 5)),
                    float(data.get("threshold", 0.5)),
                    use_igpu,
                    use_ext_env,
                    data.get("igpuEnvName") if use_ext_env else None,
                    target_chains=target_chains_val,
                    backend=data.get("backend", "auto"),
                    log_callback=logger,
                    use_npu=use_npu,
                    use_npu_ext_env=use_npu_ext_env,
                    npu_ext_env_name=(data.get("npuEnvName") if use_npu_ext_env else None),
                    use_esm=use_esm,
                    use_esm_ext_env=use_esm_ext_env,
                    esm_ext_env_name=(data.get("esmEnvName") if use_esm_ext_env else None),
                    direct_3d=use_direct_3d,
                    direct_ss_dir=direct_ss_dir,
                )
                state.set_status("completed")
                state.update_progress(100, "Completed")
                state.add_log("Job completed successfully.", "SUCCESS")
            except Exception as e:
                state.set_status("error")
                state.add_log(f"Pipeline Error: {str(e)}", "ERROR")


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
