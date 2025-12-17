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
    for record in SeqIO.parse(file_path, "fasta"):
        seq_str = str(record.seq).strip().upper()
        if not seq_str:
            head = (record.id or "").strip().upper()
            head_clean = re.sub(r"[^A-Z]", "", head)
            if len(head_clean) >= 5 and set(head_clean) <= set("ACDEFGHIKLMNPQRSTVWYXBZUO"):
                seq_str = head_clean
        sequences.append((record.id, seq_str))
        
    return sequences
