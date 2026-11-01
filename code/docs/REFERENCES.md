# References

Checked on 2026-11-01. DOI records were fetched from Crossref unless noted. Repository status
came from the GitHub API. arXiv records are preprints and have not all been peer reviewed.

## Venue
- OSDI '27 Call for Papers: https://www.usenix.org/conference/osdi27/call-for-papers
  (preliminary PDF: https://www.usenix.org/sites/default/files/osdi27_cfp_070726.pdf)

## Closest work
- Driscoll et al. Equivalence Checking of ML GPU Kernels (Volta). PACMPL 10, 2026. https://doi.org/10.1145/3839470, https://arxiv.org/abs/2511.12638
- Yang et al. Taming Bitwise Behavior in GPU Kernels with Tensor Core. arXiv 2609.11356, 2026. https://arxiv.org/abs/2609.11356
- Yao et al. TAO: Tolerance-Aware Optimistic Verification for Floating-Point Neural Networks. EuroSys 2026. https://doi.org/10.1145/3767295.3803612
- George et al. TorchLean: Formalizing Neural Networks in Lean. arXiv 2602.22631. https://arxiv.org/abs/2602.22631
- Murray. Lipschitz-Based Robustness Certification Under Floating-Point Execution. arXiv 2603.13334. https://arxiv.org/abs/2603.13334
- Browne et al. Proving LLM-Generated GPU Kernels Correct and Safe. arXiv 2610.24548. https://arxiv.org/abs/2610.24548
- Bang et al. SMT-Based Translation Validation for Machine Learning Compiler (mlir-tv). CAV 2022. https://doi.org/10.1007/978-3-031-13188-2_19

## Translation validation and tensor rewrites
- Lopes et al. Alive2: Bounded Translation Validation for LLVM. PLDI 2021. https://doi.org/10.1145/3453483.3454030
- Menendez et al. Alive-FP. SAS 2016. https://doi.org/10.1007/978-3-662-53413-7_16
- Berger et al. Translation Validation for LLVM's AArch64 Backend. OOPSLA 2025. https://2025.splashcon.org/details/OOPSLA/169/Translation-Validation-for-LLVM-s-AArch64-Backend
- Arora et al. TensorRight: Automated Verification of Tensor Graph Rewrites. POPL 2025. https://doi.org/10.1145/3704865
- Jia et al. TASO. SOSP 2019. https://doi.org/10.1145/3341301.3359630
- Yang et al. Equality Saturation for Tensor Graph Superoptimization (Tensat). arXiv 2101.01332. https://arxiv.org/abs/2101.01332
- Wang et al. PET. OSDI 2021. https://www.usenix.org/conference/osdi21/presentation/wang
- Zheng et al. EinNet. OSDI 2023. https://www.usenix.org/conference/osdi23/presentation/zheng

## Numerical error analysis
- Solovyev et al. Rigorous Estimation of Floating-Point Round-Off Errors with Symbolic Taylor Expansions (FPTaylor). TOPLAS 41, 2018. https://doi.org/10.1145/3230733
- Darulova et al. Daisy. TACAS 2018. https://doi.org/10.1007/978-3-319-89960-2_15
- Das et al. Scalable yet Rigorous Floating-Point Error Analysis (Satire). SC 2020. https://doi.org/10.1109/SC41405.2020.00055, https://arxiv.org/abs/2004.11960
- Kellison, Hsu. Numerical Fuzz. PACMPL 8 (PLDI), 2024. https://doi.org/10.1145/3656456
- Kellison et al. Bean: A Language for Backward Error Analysis. PLDI 2025. https://doi.org/10.1145/3729324
- Gappa: https://gappa.gitlabpages.inria.fr/
- PRECiSA: https://github.com/nasa/PRECiSA
- Lohar et al. Rigorous Roundoff Error Analysis of Probabilistic Floating-Point Computations. CAV 2021. https://doi.org/10.1007/978-3-030-81688-9_29
- Tao et al. Probabilistic Floating-Point Round-Off Analysis via Concentration Inequalities. arXiv 2605.04232. https://arxiv.org/abs/2605.04232
- Higham, Mary. A New Approach to Probabilistic Rounding Error Analysis. SISC 41(5), 2019. https://doi.org/10.1137/18M1226312
- Panchekha et al. Automatically Improving Accuracy for Floating Point Expressions (Herbie). PLDI 2015. https://doi.org/10.1145/2737924.2737959
- VCFloat: https://github.com/VeriNum/vcfloat. LAProof: https://github.com/VeriNum/LAProof

## Neural network verification
- Jia, Rinard. Exploiting Verified Neural Networks via Floating Point Numerical Error. SAS 2021. https://arxiv.org/abs/2003.03021
- Wang et al. Beta-CROWN. arXiv 2103.06624. https://arxiv.org/abs/2103.06624
- Paulsen et al. ReluDiff. arXiv 2001.03662. https://arxiv.org/abs/2001.03662
- Paulsen et al. NeuroDiff. arXiv 2009.09943. https://arxiv.org/abs/2009.09943
- Henzinger et al. Scalable Verification of Quantized Neural Networks. AAAI 2021. https://arxiv.org/abs/2012.08185
- VNN-COMP benchmarks: https://github.com/VNN-COMP/vnncomp2025_benchmarks

## Testing and empirical variability
- Liu et al. NNSmith. ASPLOS 2023. https://doi.org/10.1145/3575693.3575707
- Yang et al. WhiteFox. PACMPL 8 (OOPSLA2), 2024. https://doi.org/10.1145/3689736
- Wang et al. MLIRSmith. ASE 2023. https://doi.org/10.1109/ASE56229.2023.00120
- Suo et al. Fuzzing MLIR Compiler Infrastructure via Operation Dependency Analysis. ISSTA 2024. https://doi.org/10.1145/3650212.3680360
- Shen et al. A Comprehensive Study of Deep Learning Compiler Bugs. ESEC/FSE 2021. https://doi.org/10.1145/3468264.3468591
- Bentley et al. Multi-Level Analysis of Compiler-Induced Variability (FLiT). HPDC 2019. https://doi.org/10.1145/3307681.3325960
- Guo et al. pLiner. SC 2020. https://doi.org/10.1109/SC41405.2020.00053

## System documentation and source
- XLA flags, defaults: https://github.com/openxla/xla/blob/f824cdd4dce2f2c0ce092fb7cfdc7ad1ebc92163/xla/debug_options_flags.cc
- XLA contraction: https://github.com/openxla/xla/blob/f824cdd4dce2f2c0ce092fb7cfdc7ad1ebc92163/xla/service/llvm_ir/llvm_util.h
- XLA operation semantics: https://openxla.org/xla/operation_semantics
- Inductor config: https://github.com/pytorch/pytorch/blob/7b6022cde65e330461c24e5a7a18919c5aec6310/torch/_inductor/config.py
- Inductor C++ flags: https://github.com/pytorch/pytorch/blob/7b6022cde65e330461c24e5a7a18919c5aec6310/torch/_inductor/cpp_builder.py
- torch.compile: https://docs.pytorch.org/docs/stable/torch.compiler.html
- ONNX Runtime graph optimizations: https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html
- OpenVINO precision control: https://docs.openvino.ai/2025/openvino-workflow/running-inference/optimize-inference/precision-control.html
- oneDNN data types: https://uxlfoundation.github.io/oneDNN/dev_guide_data_types.html
- VDPBF16PS: https://felixcloutier.com/x86/vdpbf16ps
- MLPerf Inference: https://github.com/mlcommons/inference
