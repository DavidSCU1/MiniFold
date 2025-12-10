import os
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from modules.env_loader import load_env
from modules.input_handler import load_fasta
from modules.qwen_module import qwen_ss_candidates
from modules.llm_module import analyze_sequence, deepseek_eval_case
from modules.backbone_predictor import run_backbone_fold_multichain
from modules.visualization import generate_html_view


class MiniFoldGUI:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("MiniFold GUI")
        self.master.geometry("840x640")
        self.master.configure(bg="#f7f9fb")

        self.palette = {
            "bg": "#f7f9fb",
            "fg": "#1f2937",
            "accent": "#4a6fa5",
            "accent2": "#7aa5d2",
            "muted": "#9ca3af",
            "border": "#e5e7eb",
        }

        self.scale_var = tk.DoubleVar(value=1.0)
        self._init_style()
        self._build_layout()
        self._apply_scaling()  # ensure initial clarity提升

    def _init_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background=self.palette["bg"], foreground=self.palette["fg"], font=("Segoe UI", 10))
        style.configure("TButton", padding=6, font=("Segoe UI", 10, "bold"))
        style.configure("Accent.TButton", background=self.palette["accent"], foreground="#ffffff")
        style.map(
            "Accent.TButton",
            background=[("active", self.palette["accent2"]), ("disabled", "#cbd5e1")],
            foreground=[("disabled", "#e5e7eb")],
        )
        style.configure("TEntry", padding=4, relief="flat", fieldbackground="#ffffff")
        style.configure("TSpinbox", padding=4, relief="flat", fieldbackground="#ffffff")
        style.configure("Horizontal.TSeparator", background=self.palette["border"])

    def _build_layout(self):
        wrapper = ttk.Frame(self.master, padding=16, style="Card.TFrame")
        wrapper.pack(fill=tk.BOTH, expand=True)

        form = ttk.Frame(wrapper, padding=(8, 8, 8, 12))
        form.pack(fill=tk.X, pady=(0, 8))

        # Input file
        ttk.Label(form, text="FASTA 文件").grid(row=0, column=0, sticky="w")
        self.input_var = tk.StringVar()
        entry_in = ttk.Entry(form, textvariable=self.input_var, width=70)
        entry_in.grid(row=1, column=0, sticky="we", padx=(0, 8))
        ttk.Button(form, text="浏览", command=self._choose_fasta).grid(row=1, column=1, sticky="e")

        # Output dir
        ttk.Label(form, text="输出目录").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.out_var = tk.StringVar(value=os.path.join(os.getcwd(), "output"))
        entry_out = ttk.Entry(form, textvariable=self.out_var, width=70)
        entry_out.grid(row=3, column=0, sticky="we", padx=(0, 8))
        ttk.Button(form, text="浏览", command=self._choose_outdir).grid(row=3, column=1, sticky="e")

        # Environment
        ttk.Label(form, text="环境描述 (可选)").grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.env_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.env_var, width=70).grid(row=5, column=0, sticky="we", padx=(0, 8))

        # Params
        params = ttk.Frame(form)
        params.grid(row=6, column=0, columnspan=2, sticky="we", pady=(12, 4))
        params.columnconfigure(1, weight=1)
        params.columnconfigure(3, weight=1)

        ttk.Label(params, text="候选数量 (ssn)").grid(row=0, column=0, sticky="w")
        self.ssn_var = tk.IntVar(value=5)
        ttk.Spinbox(params, from_=1, to=10, textvariable=self.ssn_var, width=6).grid(row=0, column=1, sticky="w", padx=(4, 24))

        ttk.Label(params, text="阈值 (0-1)").grid(row=0, column=2, sticky="w")
        self.threshold_var = tk.DoubleVar(value=0.5)
        ttk.Spinbox(params, from_=0.0, to=1.0, increment=0.05, textvariable=self.threshold_var, width=6).grid(row=0, column=3, sticky="w", padx=4)

        # UI scaling for clarity
        scale_frame = ttk.Frame(form)
        scale_frame.grid(row=7, column=0, columnspan=2, sticky="we", pady=(4, 0))
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
        self.run_btn = ttk.Button(btns, text="开始运行", style="Accent.TButton", command=self._on_run)
        self.run_btn.pack(side=tk.LEFT)
        ttk.Button(btns, text="清空日志", command=self._clear_log).pack(side=tk.LEFT, padx=(8, 0))

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
        else:
            self.run_btn.state(["!disabled"])

    def _on_run(self):
        fasta = self.input_var.get().strip()
        outdir = self.out_var.get().strip()
        ssn = self.ssn_var.get()
        threshold = self.threshold_var.get()

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
            args=(fasta, outdir, self.env_var.get().strip(), ssn, threshold),
            daemon=True,
        )
        t.start()

    # Core workflow (adapted from minifold.py)
    def _run_pipeline(self, fasta, outdir, env_text, ssn, threshold):
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
                q_result = qwen_ss_candidates(sequence, env_text, num=ssn)
                cases = q_result.get("cases", [])
                cand_file = os.path.join(workdir, f"{prefix}_ss_candidates.json")
                with open(cand_file, "w", encoding="utf-8") as f:
                    import json

                    json.dump(cases, f, ensure_ascii=False, indent=2)
                raw_file = os.path.join(workdir, "raw_qwen.txt")
                with open(raw_file, "w", encoding="utf-8") as f:
                    f.write(q_result.get("raw", ""))
                self.log(f"Qwen 候选生成完成，尝试数 {q_result.get('attempts', 0)}，行数 {q_result.get('lines', 0)}。")

                kept_cases = []
                for i, case in enumerate(cases):
                    chains = case.get("chains") or []
                    if not chains:
                        continue
                    case_dir = os.path.join(workdir, f"case_{i+1}")
                    os.makedirs(case_dir, exist_ok=True)
                    chain_files = []
                    for m, ss in enumerate(chains):
                        path = os.path.join(case_dir, f"{prefix}_2_case{i+1}_{m+1}.fasta.txt")
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(ss)
                        chain_files.append(os.path.basename(path))
                    p = deepseek_eval_case(sequence, env_text, chains, req_text=req_text)
                    meta = {"case": i + 1, "p": p, "chains": len(chains), "files": chain_files}
                    meta_path = os.path.join(case_dir, f"case_{i+1}_meta.json")
                    with open(meta_path, "w", encoding="utf-8") as f:
                        import json

                        json.dump(meta, f, ensure_ascii=False, indent=2)
                    if isinstance(p, float) and p >= threshold:
                        kept_cases.append(meta)
                        self.log(f"  保留案例 {i+1}，概率 {p:.2f}")
                    else:
                        self.log(f"  丢弃案例 {i+1}，概率 {p}")
                        for fn in chain_files:
                            try:
                                os.remove(os.path.join(case_dir, fn))
                            except Exception:
                                pass
                        try:
                            os.remove(meta_path)
                        except Exception:
                            pass
                        try:
                            os.rmdir(case_dir)
                        except Exception:
                            pass

                kept_file = os.path.join(workdir, f"{prefix}_cases_kept.json")
                with open(kept_file, "w", encoding="utf-8") as f:
                    import json

                    json.dump(kept_cases, f, ensure_ascii=False, indent=2)

                annotation = analyze_sequence(sequence)
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
                        self.log(f"  构建骨架: 案例 {case_idx} (p={prob:.2f})")
                        if run_backbone_fold_multichain(sequence, chains, pdb_path):
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
                    if run_backbone_fold_multichain(sequence, [s], pdb_path):
                        html_name = f"{prefix}_{suffix}.html"
                        html_path = os.path.join(three_d_dir, html_name)
                        generate_html_view(pdb_path, html_path)
                        generated_pdbs.append({"pdb": pdb_name, "html": html_name, "chains": 1, "case": 0, "prob": 0.0})

                report_file = os.path.join(workdir, f"{prefix}_report.md")
                with open(report_file, "w", encoding="utf-8") as f:
                    f.write(f"# MiniFold Analysis Report: {seq_id}\n\n")
                    f.write("## Sequence\n")
                    f.write("```\n" + sequence + "\n```\n\n")
                    f.write("## Secondary Structure Candidates (Qwen)\n")
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

