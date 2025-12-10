import os
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
        # Clean sequence: remove whitespace, ensure uppercase
        seq_str = str(record.seq).strip().upper()
        sequences.append((record.id, seq_str))
        
    return sequences
