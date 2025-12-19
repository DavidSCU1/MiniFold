import os
import re
from Bio import SeqIO

def load_fasta(file_path):
    """
    Reads a FASTA file and returns a list of (id, sequence) tuples.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    sequences = []
    # Bio.SeqIO handles multiline fasta and cleanup
    try:
        for record in SeqIO.parse(file_path, "fasta"):
            seq_str = str(record.seq).strip().upper()
            if not seq_str:
                head = (record.id or "").strip().upper()
                head_clean = re.sub(r"[^A-Z]", "", head)
                if len(head_clean) >= 5 and set(head_clean) <= set("ACDEFGHIKLMNPQRSTVWYXBZUO"):
                    seq_str = head_clean
            if seq_str:
                sequences.append((record.id, seq_str))
    except Exception:
        pass
        
    # Fallback: If no sequences found, treat the whole file as a raw sequence
    if not sequences:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
                # Remove header-like lines if they exist but weren't parsed?
                # Just clean up non-sequence chars
                clean_seq = re.sub(r"[^A-Za-z]", "", content).upper()
                # Simple validation: mostly standard AA characters
                valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
                if len(clean_seq) > 5:
                    # Check if it looks like protein (at least 50% valid AA)
                    valid_count = sum(1 for c in clean_seq if c in valid_aa)
                    if valid_count / len(clean_seq) > 0.5:
                        sequences.append(("Raw_Sequence", clean_seq))
        except Exception:
            pass

    return sequences
