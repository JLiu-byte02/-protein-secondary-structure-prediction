# -protein-secondary-structure-prediction

Correct protein secondary structure prediction offers residue-resolution constraints useful for tertiary modeling, functional annotation, variant interpretation, and proteomic screening on the large scale. However, in the fine-grained eight-state (Q8) formulation that subdivides helices and strands, accuracy remains challenging: models must reconcile short-range sequence motifs with long-range dependencies while preserving sharp helix/strand boundaries, and reliance on multiple-sequence alignments limits scalability. We present a multi-scale, dual-stream model by utilizing alignment-free embeddings taken from pre-trained ESM protein language models and amino-acid identity to efficiently capture local sequence patterns and long-range associations while maintaining clean secondary-structure boundaries. Evaluated on CB513, TS115, TEST2018, and CASP12-13, our model achieves high accuracy in both Q8 and Q3 with consistent generalization across sets, and shows clear improvements at notoriously difficult helix/strand boundaries. A gated GRU encoder emphasizes local patterns and boundary transitions, while a bidirectional LSTM decoder captures longer-range context, a 1D atrous pyramid head fuses multi-scale evidence without degrading per-residue resolution. Our study points to a simple and scalable route to high-fidelity secondary-structure annotation by utilizing alignment-free ESM representations and contributes to downstream structural biology and protein-engineering pipelines.

Function:

Prepare secondary structure .ss files.
Run ESM_one-hot.py to convert .ss files into .feat feature files.
Run train.py to train the model on the generated .feat files.
