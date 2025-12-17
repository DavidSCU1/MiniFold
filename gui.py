import os
import threading
import traceback
import tkinter as tk
import subprocess
import sys
from tkinter import ttk, filedialog, messagebox, scrolledtext

from modules.env_loader import load_env
from modules.input_handler import load_fasta
from modules.ss_generator import pybiomed_ss_candidates
from modules.ark_module import ark_vote_cases, ark_analyze_sequence, get_default_models
from modules.backbone_predictor import run_backbone_fold_multichain
from modules.igpu_predictor import run_backbone_fold_multichain as run_igpu_fold
from modules.visualization import generate_html_view


class MiniFoldGUI:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("MiniFold GUI")
        self.master.geometry("1024x768")
        self.master.configure(bg=self.palette["bg"])

        self.palette = {
            "bg": "#fafbfc",
            "fg": "#2d3748",
            "accent": "#4299e1",
            "accent2": "#63b3ed",
            "muted": "#718096",
            "border": "#e2e8f0",
        }

        self.scale_var = tk.DoubleVar(value=1.0)
        self._init_style()
        self._build_layout()
        self._apply_scaling()  # ensure initial clarity提升

    def _init_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background=self.palette["bg"], foreground=self.palette["fg"], font=("Segoe UI", 10))
        style.configure("TButton", padding=8, font=("Segoe UI", 10))
        style.configure("Accent.TButton", background=self.palette["accent"], foreground="#ffffff")
        style.map(
            "Accent.TButton",
            background=[("active", self.palette["accent2"]), ("disabled", "#cbd5e1")],
            foreground=[("disabled", "#e5e7eb")],
        )
        style.configure("TEntry", padding=6, relief="flat", fieldbackground="#ffffff", 
                        foreground=self.palette["fg"], borderwidth=1)
        style.configure("TSpinbox", padding=6, relief="flat", fieldbackground="#ffffff",
                        foreground=self.palette["fg"], borderwidth=1)
        style.configure("Horizontal.TSeparator", background=self.palette["border"])
        style.configure("Card.TFrame", background=self.palette["bg"])
        style.configure("Title.TLabel", background=self.palette["bg"], foreground=self.palette["accent"], 
                        font=("Segoe UI", 16, "bold"))
        style.configure("Subtitle.TLabel", background=self.palette["bg"], foreground=self.palette["muted"], 
                        font=("Segoe UI", 9))

    def _build_layout(self):
        wrapper = ttk.Frame(self.master, padding=16, style="Card.TFrame")
        wrapper.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(wrapper, padding=(0, 0, 0, 8))
        header.pack(fill=tk.X)
        ttk.Label(header, text="MiniFold", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="轻量级蛋白结构分析与建模工作流", style="Subtitle.TLabel").pack(anchor="w")

        form = ttk.Frame(wrapper, padding=(8, 8, 8, 12))
        form.pack(fill=tk.X, pady=(0, 8))

        # Input file
        ttk.Label(form, text="📁 FASTA 文件").grid(row=0, column=0, sticky="w")
        self.input_var = tk.StringVar()
        entry_in = ttk.Entry(form, textvariable=self.input_var, width=70)
        entry_in.grid(row=1, column=0, sticky="we", padx=(0, 8))
        ttk.Button(form, text="浏览...", command=self._choose_fasta).grid(row=1, column=1, sticky="e")

        # Output dir
        ttk.Label(form, text="📂 输出目录").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.out_var = tk.StringVar(value=os.path.join(os.getcwd(), "output"))
        entry_out = ttk.Entry(form, textvariable=self.out_var, width=70)
        entry_out.grid(row=3, column=0, sticky="we", padx=(0, 8))
        ttk.Button(form, text="浏览...", command=self._choose_outdir).grid(row=3, column=1, sticky="e")

        # Environment
        ttk.Label(form, text="⚙️ 环境描述 (可选)").grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.env_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.env_var, width=70).grid(row=5, column=0, sticky="we", padx=(0, 8))

        # Params
        params = ttk.Frame(form)
        params.grid(row=6, column=0, columnspan=2, sticky="we", pady=(12, 4))
        params.columnconfigure(1, weight=1)
        params.columnconfigure(3, weight=1)

        ttk.Label(params, text="🎯 候选数量 (ssn)").grid(row=0, column=0, sticky="w")
        self.ssn_var = tk.IntVar(value=5)
        ttk.Spinbox(params, from_=1, to=10, textvariable=self.ssn_var, width=6).grid(row=0, column=1, sticky="w", padx=(4, 24))

        ttk.Label(params, text="📊 阈值 (0-1)").grid(row=0, column=2, sticky="w")
        self.threshold_var = tk.DoubleVar(value=0.5)
        ttk.Spinbox(params, from_=0.0, to=1.0, increment=0.05, textvariable=self.threshold_var, width=6).grid(row=0, column=3, sticky="w", padx=4)

        # iGPU Acceleration
        self.igpu_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(params, text="🚀 启用 iGPU 加速", variable=self.igpu_var).grid(row=0, column=4, sticky="w", padx=12)

        # External Environment (for iGPU)
        ext_frame = ttk.Frame(form)
        ext_frame.grid(row=7, column=0, columnspan=2, sticky="we", pady=(4, 4))
        
        self.use_ext_env_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ext_frame, text="使用独立环境 (推荐)", variable=self.use_ext_env_var).pack(side=tk.LEFT)
        
        ttk.Label(ext_frame, text="环境名称/路径:").pack(side=tk.LEFT, padx=(8, 4))
        self.ext_env_name_var = tk.StringVar(value="MiniFold_NPU")
        ttk.Entry(ext_frame, textvariable=self.ext_env_name_var, width=30).pack(side=tk.LEFT)

        # UI scaling for clarity
        scale_frame = ttk.Frame(form)
        scale_frame.grid(row=8, column=0, columnspan=2, sticky="we", pady=(4, 0))
        ttk.Label(scale_frame, text="界面缩放").grid(row=0, column=0, sticky="w")
        self.scale_label = ttk.Label(scale_frame, text="100%")
        self.scale_label.grid(row=0, column=2, sticky="e")
        scale = ttk.Scale(
            scale_frame,
            from_=0.9,
            to=1.5,
            orient="horizontal",
            variable=self.scale_var,
            command=self._on_scale_change,
        )
        scale.grid(row=0, column=1, sticky="we", padx=8)
        scale_frame.columnconfigure(1, weight=1)

        # Action buttons
        btns = ttk.Frame(wrapper)
        btns.pack(fill=tk.X, pady=(0, 8))
        self.run_btn = ttk.Button(btns, text="▶️ 开始运行", style="Accent.TButton", command=self._on_run)
        self.run_btn.pack(side=tk.LEFT)
        ttk.Button(btns, text="🗑️ 清空日志", command=self._clear_log).pack(side=tk.LEFT, padx=(8, 0))
        self.progress = ttk.Progressbar(btns, mode="indeterminate", length=160)
        self.progress.pack(side=tk.RIGHT)

        ttk.Separator(wrapper, orient="horizontal").pack(fill=tk.X, pady=6)

        # Log area
        log_frame = ttk.Frame(wrapper)
        log_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(log_frame, text="运行日志").pack(anchor="w")
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=20,
            bg="#ffffff",
            fg=self.palette["fg"],
            font=("Consolas", 10),
            relief="flat",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground=self.palette["border"],
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.log("MiniFold GUI 已就绪。请配置参数后点击开始运行。")
        status_bar = ttk.Frame(wrapper, padding=(0, 8, 0, 0))
        status_bar.pack(fill=tk.X)
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(status_bar, textvariable=self.status_var).pack(anchor="w")

    def _apply_scaling(self):
        """根据 scale_var 调整 Tk scaling 与字体大小，提升清晰度。"""
        factor = max(0.8, min(1.6, float(self.scale_var.get() or 1.0)))
        try:
            self.master.tk.call("tk", "scaling", factor)
        except tk.TclError:
            pass
        base_font = ("Segoe UI", max(9, round(10 * factor)))
        bold_font = ("Segoe UI", max(9, round(10 * factor)), "bold")
        style = ttk.Style()
        style.configure("TLabel", font=base_font)
        style.configure("TButton", font=bold_font)
        style.configure("Accent.TButton", font=bold_font)
        entry_font = ("Segoe UI", max(9, round(10 * factor)))
        for widget in self.master.winfo_children():
            self._update_widget_font(widget, entry_font)
        log_font = ("Consolas", max(9, round(10 * factor)))
        self.log_text.configure(font=log_font)
        self.scale_label.configure(text=f"{int(factor * 100)}%")

    def _update_widget_font(self, widget, font):
        if isinstance(widget, (ttk.Entry, ttk.Spinbox)):
            widget.configure(font=font)
        elif isinstance(widget, tk.Text):
            widget.configure(font=font)
        if isinstance(widget, (ttk.Frame, tk.Frame)):
            for child in widget.winfo_children():
                self._update_widget_font(child, font)

    def _on_scale_change(self, *_):
        self._apply_scaling()

    # UI helpers
    def _choose_fasta(self):
        path = filedialog.askopenfilename(filetypes=[("FASTA", "*.fasta *.fa *.faa *.fsa *.txt"), ("All", "*.*")])
        if path:
            self.input_var.set(path)

    def _choose_outdir(self):
        path = filedialog.askdirectory()
        if path:
            self.out_var.set(path)

    def log(self, msg: str):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.master.update_idletasks()

    def _clear_log(self):
        self.log_text.delete("1.0", tk.END)

    def _set_running(self, running: bool):
        if running:
            self.run_btn.state(["disabled"])
            try:
                self.progress.start(12)
            except Exception:
                pass
            self.status_var.set("运行中")
        else:
            self.run_btn.state(["!disabled"])
            try:
                self.progress.stop()
            except Exception:
                pass
            self.status_var.set("就绪")

    def _on_run(self):
        fasta = self.input_var.get().strip()
        outdir = self.out_var.get().strip()
        ssn = self.ssn_var.get()
        threshold = self.threshold_var.get()
        use_igpu = self.igpu_var.get()
        use_ext_env = self.use_ext_env_var.get()
        ext_env_name = self.ext_env_name_var.get().strip()

        if not fasta:
            messagebox.showwarning("提示", "请先选择 FASTA 文件。")
            return
        if not os.path.exists(fasta):
            messagebox.showerror("错误", "FASTA 文件不存在。")
            return
        if ssn < 1:
            messagebox.showwarning("提示", "候选数量必须 >= 1。")
            return
        if threshold < 0.0 or threshold > 1.0:
            messagebox.showwarning("提示", "阈值需在 0-1 之间。")
            return
        if not outdir:
            outdir = os.path.join(os.getcwd(), "output")
            self.out_var.set(outdir)

        self._set_running(True)
        self.log("==== 开始运行 MiniFold 工作流 ====")
        t = threading.Thread(
            target=self._run_pipeline,
            args=(fasta, outdir, self.env_var.get().strip(), ssn, threshold, use_igpu, use_ext_env, ext_env_name),
            daemon=True,
        )
        t.start()

    # Core workflow (adapted from minifold.py)
    def _run_pipeline(self, fasta, outdir, env_text, ssn, threshold, use_igpu, use_ext_env, ext_env_name):
        try:
            load_env()
            os.makedirs(outdir, exist_ok=True)
            self.log(f"读取 FASTA: {fasta}")
            sequences = load_fasta(fasta)
            if not sequences:
                self.log("未发现有效序列。")
                return

            input_base = os.path.basename(fasta)
            prefix = os.path.splitext(input_base)[0]
            workdir = os.path.join(outdir, prefix)
            three_d_dir = os.path.join(workdir, "3d_structures")
            os.makedirs(three_d_dir, exist_ok=True)

            req_path = os.path.join(workdir, "requirements.txt")
            req_lines = [
                f"env={env_text}",
                f"ssn={ssn}",
                f"threshold={threshold}",
            ]
            os.makedirs(workdir, exist_ok=True)
            with open(req_path, "w", encoding="utf-8") as f:
                f.write("\n".join(req_lines))
            with open(req_path, "r", encoding="utf-8") as f:
                req_text = f.read()

            for seq_id, sequence in sequences:
                self.log(f"处理序列: {seq_id} (长度 {len(sequence)})")
                
                # 1. Generate Candidates (PyBioMed)
                q_result = pybiomed_ss_candidates(sequence, env_text, num=ssn)
                cases = q_result.get("cases", [])
                cand_file = os.path.join(workdir, f"{prefix}_ss_candidates.json")
                with open(cand_file, "w", encoding="utf-8") as f:
                    import json
                    json.dump(cases, f, ensure_ascii=False, indent=2)
                raw_file = os.path.join(workdir, "raw_candidates.txt")
                with open(raw_file, "w", encoding="utf-8") as f:
                    f.write(q_result.get("raw", ""))
                self.log(f"候选生成完成，尝试数 {q_result.get('attempts', 0)}，行数 {q_result.get('lines', 0)}。")

                # 2. Verify (Ark Voting)
                self.log("正在进行 Ark 多模型投票验证...")
                models = get_default_models()
                votes = ark_vote_cases(models, sequence, env_text, cases, req_text=req_text)
                
                votes_file = os.path.join(workdir, f"{prefix}_votes.json")
                with open(votes_file, "w", encoding="utf-8") as f:
                    import json
                    json.dump(votes, f, ensure_ascii=False, indent=2)

                kept_cases = []
                if votes.get("cases"):
                    for case_it in votes["cases"]:
                        avg_score = case_it.get("avg", 0.0)
                        idx = case_it.get("idx")
                        
                        # Prepare files for this case
                        case_data = cases[idx]
                        chains = case_data.get("chains", [])
                        if not chains: continue
                        
                        case_dir = os.path.join(workdir, f"case_{idx+1}")
                        os.makedirs(case_dir, exist_ok=True)
                        chain_files = []
                        for m, ss in enumerate(chains):
                            path = os.path.join(case_dir, f"{prefix}_2_case{idx+1}_{m+1}.fasta.txt")
                            with open(path, "w", encoding="utf-8") as f:
                                f.write(ss)
                            chain_files.append(os.path.basename(path))
                            
                        meta = {
                            "case": idx + 1, 
                            "p": avg_score, 
                            "chains": len(chains), 
                            "files": chain_files
                        }
                        
                        if avg_score >= threshold:
                            kept_cases.append(meta)
                            self.log(f"  保留案例 {idx+1}，评分 {avg_score:.2f}")
                        else:
                            self.log(f"  丢弃案例 {idx+1}，评分 {avg_score:.2f}")
                            # Clean up
                            for fn in chain_files:
                                try: os.remove(os.path.join(case_dir, fn))
                                except: pass
                            try: 
                                os.rmdir(case_dir)
                            except: 
                                pass

                kept_file = os.path.join(workdir, f"{prefix}_cases_kept.json")
                with open(kept_file, "w", encoding="utf-8") as f:
                    import json
                    json.dump(kept_cases, f, ensure_ascii=False, indent=2)

                self.log("候选验证完成。正在分析序列功能...")
                # 3. Annotation
                annotation = ark_analyze_sequence(sequence)
                annotation_file = os.path.join(workdir, f"{prefix}_annotation.txt")
                with open(annotation_file, "w", encoding="utf-8") as f:
                    f.write(annotation)
                self.log("功能注释完成。")

                generated_pdbs = []
                self.log("生成 3D 结构...")
                if kept_cases:
                    for rank, meta in enumerate(sorted(kept_cases, key=lambda x: x.get("p", 0.0), reverse=True), start=1):
                        prob = meta.get("p", 0.0)
                        case_idx = meta.get("case")
                        case_dir = os.path.join(workdir, f"case_{case_idx}")
                        chains = []
                    for fn in meta.get("files", []):
                        with open(os.path.join(case_dir, fn), "r", encoding="utf-8") as f:
                            chains.append(f.read().strip())
                    suffix = f"case{case_idx}_model_{rank}"
                    pdb_name = f"{prefix}_{suffix}.pdb"
                    pdb_path = os.path.join(three_d_dir, pdb_name)
                    
                    if use_igpu:
                        if use_ext_env and ext_env_name:
                            self.log(f"  > Case {case_idx} (p={prob:.2f}): Optimizing backbone (iGPU via external env '{ext_env_name}')...")
                            import json
                            igpu_input_data = {"sequence": sequence, "chains": chains}
                            tmp_input = os.path.join(workdir, f"igpu_input_case{case_idx}.json")
                            with open(tmp_input, "w", encoding="utf-8") as f:
                                json.dump(igpu_input_data, f)
                            
                            runner_script = os.path.join(os.getcwd(), "modules", "igpu_runner.py")
                            if os.path.sep in ext_env_name or "/" in ext_env_name:
                                # Path to python executable
                                cmd_list = [ext_env_name, runner_script, "--input", tmp_input, "--output", pdb_path]
                                cmd_str = subprocess.list2cmdline(cmd_list)
                            else:
                                # Conda env name
                                cmd_list = ["conda", "run", "-n", ext_env_name, "python", runner_script, "--input", tmp_input, "--output", pdb_path]
                                cmd_str = f'conda run -n {ext_env_name} python "{runner_script}" --input "{tmp_input}" --output "{pdb_path}"'
                            
                            try:
                                # Use shell=True for conda on Windows
                                result = subprocess.run(cmd_str if os.name == "nt" else cmd_list, 
                                                        capture_output=True, 
                                                        text=True, 
                                                        encoding="utf-8", 
                                                        shell=(os.name=="nt"))
                                
                                # Log iGPU Runner output to GUI log for visibility
                                if result.stdout:
                                    self.log(f"[iGPU Output] {result.stdout.strip()}")
                                if result.stderr:
                                    self.log(f"[iGPU Error] {result.stderr.strip()}")
                                    
                                if result.returncode == 0:
                                    success = True
                                else:
                                    success = False
                                    self.log(f"    iGPU External Failed (RC={result.returncode})")
                            except Exception as e:
                                success = False
                                self.log(f"    Execution Error: {e}")
                            
                            if os.path.exists(tmp_input):
                                try: os.remove(tmp_input) 
                                except: pass
                        else:
                            self.log(f"  > Case {case_idx} (p={prob:.2f}): Optimizing backbone (iGPU)...")
                            success = run_igpu_fold(sequence, chains, pdb_path)
                    else:
                        self.log(f"  > Case {case_idx} (p={prob:.2f}): Optimizing backbone (Standard)...")
                        success = run_backbone_fold_multichain(sequence, chains, pdb_path)

                    if success:
                        html_name = f"{prefix}_{suffix}.html"
                        html_path = os.path.join(three_d_dir, html_name)
                        generate_html_view(pdb_path, html_path)
                        generated_pdbs.append({"pdb": pdb_name, "html": html_name, "chains": len(chains), "case": case_idx, "prob": prob})
                else:
                    self.log("无保留案例，使用回退模型。")
                    L = len(sequence)
                    pattern = "HHHHH" + "CC" + "EEEEE" + "C"
                    s = (pattern * ((L // len(pattern)) + 1))[:L]
                    suffix = "fallback_model"
                    pdb_name = f"{prefix}_{suffix}.pdb"
                    pdb_path = os.path.join(three_d_dir, pdb_name)
                    
                    success = False
                    if use_igpu:
                        if use_ext_env and ext_env_name:
                            self.log(f"  > Fallback: Optimizing backbone (iGPU via external env)...")
                            import json
                            igpu_input_data = {"sequence": sequence, "chains": [s]}
                            tmp_input = os.path.join(workdir, f"igpu_input_fallback.json")
                            with open(tmp_input, "w", encoding="utf-8") as f:
                                json.dump(igpu_input_data, f)
                            
                            runner_script = os.path.join(os.getcwd(), "modules", "igpu_runner.py")
                            if os.path.sep in ext_env_name or "/" in ext_env_name:
                                cmd_list = [ext_env_name, runner_script, "--input", tmp_input, "--output", pdb_path]
                                cmd_str = subprocess.list2cmdline(cmd_list)
                            else:
                                cmd_list = ["conda", "run", "-n", ext_env_name, "python", runner_script, "--input", tmp_input, "--output", pdb_path]
                                cmd_str = f'conda run -n {ext_env_name} python "{runner_script}" --input "{tmp_input}" --output "{pdb_path}"'
                            
                            try:
                                result = subprocess.run(cmd_str if os.name == "nt" else cmd_list, 
                                                        capture_output=True, 
                                                        text=True, 
                                                        encoding="utf-8", 
                                                        shell=(os.name=="nt"))
                                
                                # Log iGPU Runner output to GUI log for visibility
                                if result.stdout:
                                    self.log(f"[iGPU Output] {result.stdout.strip()}")
                                if result.stderr:
                                    self.log(f"[iGPU Error] {result.stderr.strip()}")

                                if result.returncode == 0:
                                    success = True
                                else:
                                    success = False
                                    self.log(f"    iGPU External Failed (RC={result.returncode})")
                            except Exception as e:
                                success = False
                                self.log(f"    Execution Error: {e}")
                            
                            if os.path.exists(tmp_input):
                                try: os.remove(tmp_input) 
                                except: pass
                        else:
                            success = run_igpu_fold(sequence, [s], pdb_path)
                    else:
                        success = run_backbone_fold_multichain(sequence, [s], pdb_path)

                    if success:
                        html_name = f"{prefix}_{suffix}.html"
                        html_path = os.path.join(three_d_dir, html_name)
                        generate_html_view(pdb_path, html_path)
                        generated_pdbs.append({"pdb": pdb_name, "html": html_name, "chains": 1, "case": 0, "prob": 0.0})

                report_file = os.path.join(workdir, f"{prefix}_report.md")
                with open(report_file, "w", encoding="utf-8") as f:
                    f.write(f"# MiniFold Analysis Report: {seq_id}\n\n")
                    f.write("## Sequence\n")
                    f.write("```\n" + sequence + "\n```\n\n")
                    f.write("## Secondary Structure Candidates (PyBioMed)\n")
                    f.write(f"Saved: `{os.path.basename(cand_file)}`\n")
                    f.write(f"Raw: `{os.path.basename(raw_file)}`\n\n")
                    f.write("## Filtered by Qwen (Cases)\n")
                    f.write(f"Saved: `{os.path.basename(kept_file)}`\n\n")
                    f.write("## Functional Annotation (DeepSeek V3.2)\n")
                    f.write(annotation + "\n\n")
                    f.write("## 3D Models (Backbone Predictor, Multichain)\n")
                    if generated_pdbs:
                        for m in generated_pdbs:
                            f.write(f"### Case {m['case']} (Likelihood: {m['prob']:.2f})\n")
                            f.write(f"- Chains: {m['chains']}\n")
                            f.write(f"- PDB: `{m['pdb']}`\n")
                            f.write(f"- View: [Interactive 3D]({m['html']})\n\n")
                    else:
                        f.write("No models generated.\n")

                log_path = os.path.join(workdir, "process_report.log")
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(f"sequences={len(sequences)}\n")
                    f.write(f"cases={len(cases)}\n")
                    f.write(f"cases_kept={len(kept_cases)}\n")
                    f.write(f"pdb_generated={len(generated_pdbs)}\n")
                    f.write(f"qwen_attempts={q_result.get('attempts', 0)}\n")
                    f.write(f"qwen_raw_lines={q_result.get('lines', 0)}\n")

                self.log(f"报告已生成: {report_file}")
                self.log(f"PDB/HTML 输出目录: {three_d_dir}")

            self.log("==== 运行完成 ====")
            messagebox.showinfo("完成", "MiniFold 运行完成，可查看输出目录。")
        except Exception as e:
            self.log("运行失败：")
            self.log(str(e))
            self.log(traceback.format_exc())
            messagebox.showerror("错误", f"运行失败：{e}")
        finally:
            self._set_running(False)


def main():
    root = tk.Tk()
    app = MiniFoldGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

