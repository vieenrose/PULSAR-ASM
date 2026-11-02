# Nemotron RE-USE CUDA engine (planned)

S2-S6: single-TU CUDA sm_53 engine. No cuFFT, no cuBLAS.
- STFT/iSTFT via DFT-GEMM
- DenseEncoder, 30× TFMamba (fused selective scan)
- Mag/Phase decoders
- Full-clip + chunked (5s, 50% hop) modes
