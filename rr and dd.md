Phase 1 Publishing R&D Roadmap: Adversarial Scrutiny and Experimental Redesign of the EQARNB Architecture
Executive Overview of Project Integrity
The investigation into the EQARNB (Equivariant Q-Attention with a Thermodynamic Gate and Stiefel-constrained causal layer) architecture represents a highly ambitious intersection of multimodal physiological sensing, representation learning, and algorithmic fairness. The core thesis posits that a Stiefel-constrained causal layer can dynamically suppress a highly correlated synthetic visual confounder—specifically, a rendered facial scar with a correlation coefficient of ρ=0.85 to the target label—while preserving multimodal stress classification accuracy across physiological and visual branches. Theoretically, this hypothesis navigates the complex problem of shortcut learning in deep neural networks, attempting to force a model to learn causal physiological features rather than relying on superficial, biased correlations.

However, a rigorous adversarial review of the repository in its currently locked, finalized thesis state reveals that the empirical claims cannot withstand peer scrutiny at the level expected by top-tier venues such as CVPR, NeurIPS, or IEEE T-BME. The project is fundamentally compromised by a series of fatal methodological flaws, statistical fragility, and severe data leakage. The research is bifurcated into two distinct trajectories: Line A (The UBFC-Phys Leakage Analysis) and Line B (The CCB Benchmark). Both lines exhibit critical vulnerabilities that invalidate their current findings.

Line A relies on a critically underpowered sample size of N=4 subjects and a statistically fragile temporal smoothing parameter to assert a negative result regarding the impact of video compression on Remote Photoplethysmography (rPPG) signals. Line B is entirely invalidated by the "Stranger-Pairing Flaw," an experimental design failure wherein the dataset pairs unrelated facial images with independent physiological data. This pairing destroys all true mutual information in the visual branch, reducing the proposed confounder suppression to trivial artifact detection.   

To elevate this project to the standards of rigorous scientific publication, the experimental design must be completely overhauled. This report provides an exhaustive adversarial deconstruction of the known flaws, a mathematical verification of the underlying architectural components, and a strict Phase 1 Research and Development (R&D) roadmap. This roadmap is designed to guide the local engineering AI (the Antigravity CLI Agent) in eradicating the repository's technical debt through precise, empirical redesign.

Section 1: Architectural Verification and Theoretical Deconstruction
Before addressing the empirical flaws in the data pipeline, it is necessary to rigorously verify the theoretical underpinnings of the EQARNB architecture. The model relies on the integration of video-based rPPG extraction, Equivariant Q-Attention for feature representation, and a Stiefel-constrained causal layer optimized via Riemannian geometry to enforce feature disentanglement. The mathematical foundations of these components are sound, but their current implementation is rendered untestable by the surrounding data infrastructure.

The Stiefel Manifold and Riemannian Optimization
The core of the causal layer relies on maintaining strict orthonormality in its parameter matrices. This constraint is intended to prevent gradient collapse, stabilize training dynamics, and enforce independent, disentangled feature representations that are robust to spurious correlations. The parameter matrices W∈R 
d×r
  (where d≥r) are constrained to the Stiefel manifold, a smooth Riemannian manifold defined mathematically as St(d,r)={W∈R 
d×r
 :W 
T
 W=I 
r
​
 }.   

Standard optimization algorithms, such as Euclidean Stochastic Gradient Descent (SGD) or Adam, cannot be directly applied to these parameter matrices because standard gradient updates do not respect the curvature of the manifold, inevitably pulling the weights off the Stiefel manifold and destroying the orthogonality constraint. Therefore, Riemannian optimization is strictly required.   

The standard approach for updating parameters on a Riemannian manifold involves projecting the Euclidean gradient onto the tangent space of the manifold, computing the update step, and then using a retraction map to pull the updated point back onto the manifold surface. The exact exponential map on the Stiefel manifold is computationally prohibitive for deep neural networks due to the requirement of matrix inversion or singular value decomposition (SVD) at every training step. The EQARNB architecture correctly identifies this bottleneck and employs a retraction map based on the Cayley transform.   

For a skew-symmetric matrix A=GW 
T
 −WG 
T
  (where G is the projected Euclidean gradient), the closed-form Cayley transform is given by:

Y(α)=(I− 
2
α
​
 A) 
−1
 (I+ 
2
α
​
 A)W
Because the matrix inversion operation O(d 
3
 ) remains computationally expensive for large layers, the architecture must utilize an iterative approximation of the Cayley transform to maintain computational feasibility during training:

Y(α)=W+ 
2
α
​
 A(W+Y(α))
This fixed-point iterative formulation allows the network to implicitly transport vectors across the tangent space of the manifold using only matrix multiplications, ensuring that the orthonormality constraint is strictly enforced without catastrophic computational overhead. The theoretical application of this geometry to the causal layer is elegant. However, the efficacy of this constraint in EQARNB is currently untestable because the data fed into the layer (specifically the multimodal vision branch) contains no valid signal to disentangle—a fatal flaw that will be dissected extensively in Section 3.   

Remote Photoplethysmography (rPPG) Algorithmic Theory
The vision branch of the architecture depends heavily on the assumption that subtle chromatic variations in facial video, induced by the cardiac cycle, can be reliably extracted to form a Blood Volume Pulse (BVP) signal. The underlying principle, modeled by the dichromatic reflection model, states that light reflecting off human skin consists of a specular component (surface reflection of the illuminant, containing no physiological information) and a diffuse component (light scattered within the epidermal and dermal layers, modulated by the absorption spectra of oxygenated hemoglobin).   

The thesis evaluates three core algorithmic estimators for this extraction: POS (Plane-Orthogonal-to-Skin), CHROM (Chrominance-based), and PBV (Blood Volume Pulse Signature).

rPPG Algorithm	Theoretical Mechanism	Strengths	Vulnerabilities
CHROM	
Projects raw RGB signals into two orthogonal chrominance vectors (X 
CHROM
​
  and Y 
CHROM
​
 ) to mathematically eliminate the additive specular reflection component, assuming a standardized skin color for white-balancing.

Highly robust against motion artifacts that induce specular reflection changes.	Performance degrades under severe illumination changes or aggressive spatial subsampling.
POS	
Defines a projection plane orthogonal to the temporally normalized skin tone direction in the RGB space. It creates a projection vector z to estimate the pulse signal p(t) while minimizing motion-induced intensity variations across all three channels simultaneously.

Skin-tone independent; dynamically adapts to varying luminance spectra.	Susceptible to high-frequency temporal noise if not adequately band-pass filtered.
PBV	
Utilizes the explicit physiological signature of blood volume changes across different wavelengths to mathematically distinguish the pulse-induced color changes from broad-spectrum motion noise.

Provides strong theoretical guarantees under controlled lighting conditions.	Requires highly accurate initial calibration of the blood volume signature matrix.
  
These algorithms operate optimally on uncompressed, high-fidelity video streams where the subtle color variations are preserved. Line A of the project attempts to use these algorithms to prove that lossy video compression acts as a natural adversarial filter against rPPG leakage. This assertion entirely collapses under empirical review.   

Section 2: Critical Deconstruction of Line A (UBFC-Phys Leakage Analysis)
Line A attempts to formally prove that aggressive video compression (specifically H.264 encoding) destroys recoverable rPPG signals. If true, this would validate the premise that multimodal models utilizing heavily compressed facial video are not "cheating" by reading the pulse directly from the face, but are instead relying on higher-order facial expressions or micro-gestures. The thesis reports a negative result: compression does not reliably destroy the signal. While this finding aligns with established literature demonstrating that BVP signals can survive substantial bit rate reductions and quantization parameter adjustments, the methodology used to arrive at this conclusion in the current repository is fundamentally broken and cannot be published.   

The N=4 Statistical Power Crisis
The most glaring and immediate flaw in Line A is the reliance on a critically underpowered sample size. The analysis is currently restricted to N=4 subjects, comprising only 12 video clips. This sample size is statistically invalid for drawing broad conclusions about algorithmic robustness across human populations.

The dataset utilized for this analysis is UBFC-Phys, a rigorously designed multimodal dataset created specifically for psychophysiological studies of social stress. The full UBFC-Phys corpus contains high-resolution video recordings, contact BVP (via Empatica E4 wristbands), and Electrodermal Activity (EDA) for 56 distinct subjects. The subjects were exposed to a modified Trier Social Stress Test (TSST) consisting of a relaxation mode, a speech task (simulating a job interview with social-evaluative threat), and a mental arithmetic task.   

Evaluating the efficacy of rPPG extraction under varying compression regimes on merely 4 subjects out of an available 56 renders any resulting p-values or confidence intervals statistically meaningless. Inter-subject variability in skin tone (Fitzpatrick scale), facial hemodynamics, resting heart rate, and subtle head motion requires robust validation across a diverse cohort. State-of-the-art rPPG benchmarks mandate evaluations across the entire available corpus using Leave-One-Subject-Out (LOSO) cross-validation to account for this inherent biological variance. A sample size of N=4 is highly susceptible to outlier skew; a single subject with uniquely clean hemodynamics or minimal head motion will drastically distort the mean error rates of the entire study.   

Video Compression and the O7 Fragility Finding
Video compression algorithms, such as H.264 (Advanced Video Coding) and H.265 (High-Efficiency Video Coding), are designed to exploit spatial and temporal redundancies in visual data to reduce file size. They utilize block-based motion compensation, transforming groups of pixels into the frequency domain via the Discrete Cosine Transform (DCT) and aggressively quantizing high-frequency details that are deemed visually imperceptible to the human eye. Because rPPG relies on variations that are often sub-pixel or sub-perceptual, compression artifacts (like chroma subsampling or macroblock boundary distortion) are theoretically highly detrimental to signal recovery.   

The statistical fragility of Line A's attempt to measure this detriment is fully exposed by the "O7 Fragility Finding." The experimental design tested the recovery of the rPPG signal under different temporal smoothing configurations, specifically an Exponential Moving Average (EMA) with α=1.0 (no smoothing) and α=0.5 (heavy smoothing).

Initially, the α=1.0 arm yielded a statistically significant result indicating that the signal was recovered despite compression. However, the removal of just two out of the twelve clips—reportedly due to bounding box anomalies during face detection—caused the p-value of this arm to spike to p=0.0857, crossing the threshold into non-significance. Consequently, the primary thesis claim now rests entirely on the α=0.5 arm.

This represents a classic example of analytical cherry-picking and statistical masking. Heavy temporal smoothing artificially inflates the apparent temporal coherence of the extracted rPPG signal by suppressing high-frequency noise; it acts as an aggressive low-pass filter. This masking effect conflates the actual survival of the physiological pulse with the algorithmic smoothing artifact generated by the EMA. Without a rigorous, untampered ablation study comparing raw Signal-to-Noise Ratio (SNR) and Mean Absolute Error (MAE) across multiple, defined compression regimes (e.g., varying the Constant Rate Factor, CRF, in H.264 from 18 to 36) without EMA interference, the results are artifactual and scientifically unpublishable.   

The Chain of Custody Break in Spatial Averaging
Beyond the statistical failures, Line A suffers from a severe software engineering failure that compromises the scientific reproducibility of the entire facial cropping pipeline. Before color signals can be extracted, the region of interest (ROI)—typically the cheeks and forehead—must be accurately bounded and spatially averaged to reduce camera sensor noise.   

The repository's core spatial averaging crop rule relies on the mathematical calculation w = IOD \times 3.566283 \times 1.5, where IOD is the Inter-Ocular Distance. The constant 3.566283 is hardcoded into the pipeline, and the original script utilized to generate this constant has been permanently overwritten.

In modern computer vision pipelines utilizing the MediaPipe Face Landmarker, the 3D face mesh provides 478 distinct, highly precise facial landmarks. The IOD is typically defined as the Euclidean distance between the pupils (MediaPipe indices 468 and 473) or the outer eye corners (MediaPipe indices 33 and 263). While the population-mean IOD in human adults is approximately 62 mm, translating this to pixel space requires dynamic calculation based on camera focal length and subject distance.   

Utilizing a hardcoded "magic number" without an explicit, reproducible mathematical derivation from the raw Cartesian coordinates of the source landmarks violates the fundamental principle of scientific replicability. This constant cannot be defended in peer review; it must be empirically re-derived from the source tensors to restore the chain of custody.

Section 3: Dissection of Line B (CCB Benchmark and Confounder Suppression)
Line B represents the core architectural contribution of the thesis: the training of the EQARNB network to suppress a synthetic visual confounder (a scar) while maintaining acute stress classification accuracy. While Line B was correctly excluded from the formalized thesis document due to recognized flaws, rescuing this benchmark requires a complete paradigm shift. In its current state, Line B is scientifically invalid due to a cascading series of fatal experimental design failures that render its output meaningless.

The Stranger-Pairing Flaw and Zero Mutual Information
The most catastrophic error in the entire repository—an error that fundamentally destroys the validity of the multimodal fusion architecture—is the "Stranger-Pairing Flaw." The CCB Benchmark constructs its multimodal inputs by arbitrarily pairing high-resolution facial images from the FFHQ (Flickr-Faces-HQ) dataset (Subject A) with ground-truth physiological signals (ECG, BVP, EDA) from the WESAD (Wearable Stress and Affect Detection) dataset (Subject B).

In a legitimate multimodal stress detection task, the vision branch relies on analyzing genuine physiological and behavioral cues embedded in the face of the subject experiencing the stress. These include micro-expressions, pupillary dilation, blink rate variability, and rPPG chromatic variations. Simultaneously, the physiological branch relies on Heart Rate Variability (HRV) metrics extracted from ECG or PPG data recorded from the same subject at the same time. Key HRV features widely utilized in WESAD benchmarking include the Root Mean Square of Successive Differences (RMSSD), the Standard Deviation of NN intervals (SDNN), and the Low-Frequency to High-Frequency (LF/HF) power ratio, all of which exhibit well-documented shifts under acute sympathetic nervous system arousal.   

By pairing an isolated, static FFHQ face (which contains no stress state context) with dynamic WESAD physiology, the mutual information between the visual modality and the true physiological stress state label is forced to exactly zero (I(X 
vision
​
 ;Y)=0). There is absolutely no learnable facial feature that correlates with the WESAD stress label.

To create the targeted "bias," the researchers planted a synthetic, rendered scar on a subset of the FFHQ faces, engineering a Pearson correlation of ρ=0.85 between the presence of the scar and the positive stress label. Because the scar is the only visual feature in the entire image tensor that carries any predictive signal for the label, the EQARNB architecture is not actually suppressing a confounder while balancing true visual features. It is simply learning to identify a synthetic tag (shortcut learning) and, upon restriction by the causal layer, zeroing out the entire vision branch to minimize its overall loss function. This is a trivial manifestation of neural network laziness, not a demonstration of advanced causal debiasing.   

The Illusion of the Demographic Parity Gap Collapse
The project attempts to evaluate the success of its fairness constraints and confounder suppression using Demographic Parity (DP). DP is a standard fairness metric requiring that the model's positive prediction rate is statistically independent of the sensitive attribute (in this case, the synthetic scar). It is expressed mathematically as P( 
Y
^
 =1∣A=1)=P( 
Y
^
 =1∣A=0), meaning the model should predict stress at equal rates regardless of whether the subject has a scar.   

The repository touts a massive collapse in the DP gap—from an initial highly biased 0.73 down to a fair 0.004—as absolute proof of the Stiefel-constrained causal layer's efficacy. Under adversarial scrutiny, this metric is entirely circular and meaningless.

Because of the Stranger-Pairing Flaw, the causal layer easily identifies during training that the vision branch contains no robust predictive power other than the highly correlated scar. When the architecture applies the thermodynamic gate and the orthogonality constraint on the Stiefel manifold, it does not refine the visual features; it completely amputates the visual modality. The model pivots to relying 100% on the WESAD physiological signals (which contain the true label) to make the classification. A DP gap of 0.004 is trivial to achieve if the model simply ignores the image completely. This is not algorithmic fairness or causal debiasing; it is modality collapse masked by misleading metrics.   

Missing Regimes, Baselines, and Controls
The experimental matrix for Line B is heavily impoverished. It lacks the fundamental controls and baselines required to contextualize the network's behavior and prove that multimodal fusion is occurring.

Missing Component	Scientific Purpose	Consequence of Omission
Robustness Regimes	
The model was only evaluated under extreme bias (ρ=0.85). Rigorous evaluation of causal debiasing requires mapping the model's performance decay across multiple bias regimes, specifically ρ=0.5 (moderate correlation) and ρ=0.15 (near-random correlation).

Inability to prove that the causal layer scales adaptively to varying levels of dataset contamination.
Sham Edits	The rendering of a synthetic scar introduces high-frequency visual artifacts and localized pixel distortion into the FFHQ image. A "Sham Edit" (e.g., an imperceptible noise patch or a neutral geometry modification applied with the same rendering pipeline but zero label correlation) must be introduced.	It is impossible to verify whether the convolutional backbone of the EQARNB model is reacting to the semantic concept of a "scar" or merely the localized artifact of digital alteration.
K1 Baseline (Physiology-Only)	A baseline trained exclusively on WESAD HRV features (RMSSD, LF/HF, etc.) mapping directly to the stress label, bypassing the vision branch entirely.	Fails to establish the absolute performance ceiling of the physiological modality, making it impossible to measure whether EQARNB actually improves accuracy through multimodal fusion.
K1b Baseline (Balanced ERM)	A standard Empirical Risk Minimization (ERM) multimodal baseline trained without the EQARNB causal layer.	The exact extent of the baseline shortcut learning cannot be quantified, rendering the comparison of the causal layer's efficacy void.
  
The Contaminated Legacy Checkpoints
Finally, the repository contains a critical software artifact: 24 legacy checkpoints generated prior to the resolution of the "Sentinel Bug" in the train_cgf_fair.py script. This specific bug caused degenerate epochs (yielding a hardcoded −999.0 score due to gradient explosion or NaN loss) to erroneously overwrite the best historical model checkpoints. While the underlying Python code has since been patched, the data artifacts remain in the file system. Any analytical claims, visualizations, or t-SNE projections resting on these 24 contaminated checkpoints are fundamentally poisoned. They must be immediately deprecated and isolated from the repository's active evaluation suite to prevent accidental inclusion in future analysis.

Section 4: Phase 1 Publishing R&D Roadmap
To successfully transition this thesis from a fatally flawed local repository into a rigorous, peer-reviewed publication suitable for a top-tier venue, the immediate focus must shift away from iterative algorithmic tweaking. The project requires fundamental data generation redesign and massive statistical scaling. The Phase 1 R&D Roadmap is divided into two strict trajectories: the statistical remediation of the Line A leakage analysis, and the foundational rebuilding of the Line B multimodal benchmark.

Trajectory A: Scaling and Rigorous Verification in Leakage Analysis
Line A does not require a redesign of its core architecture, but it requires a massive escalation in statistical power and procedural rigor to achieve publication grade.

Scale to N=56 Subjects: The experimental pipeline must be executed across the entire available UBFC-Phys dataset. Extracting POS, CHROM, and PBV signals under varying H.264 compression bitrates across all 56 subjects will provide a definitive, unassailable statistical foundation.   

Implementation of LOSO Cross-Validation: The pipeline must enforce a strict Leave-One-Subject-Out (LOSO) cross-validation protocol. Evaluating the SNR degradation under compression across 55 training subjects and 1 held-out test subject iteratively ensures that the findings are robust to inter-subject variability in skin tone and hemodynamics.   

Ablation of Temporal Smoothing (α): The masking effect of the Exponential Moving Average must be explicitly quantified and isolated. The evaluation protocol must report the raw SNR and Mean Absolute Error (MAE) for heart rate estimation across a matrix of smoothing parameters: α∈{1.0,0.75,0.5,0.25}. This will definitively separate the survival of the physiological pulse from the algorithmic smoothing artifact.   

Mathematical Re-derivation of the IOD Constant: The hardcoded 3.566283 multiplier must be mathematically reverse-engineered. This requires writing an independent script to extract the raw Cartesian coordinates from the MediaPipe FaceMesh (specifically isolating the outer canthi at indices 33 and 263, or the pupillary coordinates at indices 468 and 473) across a control set of frames, and dynamically calculating the required scaling ratio to restore the chain of custody.   

Trajectory B: Eradicating the Stranger-Pairing Flaw
Line B requires a complete, scorched-earth data generation redesign. The current FFHQ-WESAD paired dataset must be discarded immediately.

Transition to Native Multimodal Datasets: To prove true causal confounder suppression, the vision branch must contain valid, non-synthetic label signals that correlate with the physiological ground truth. The model must be trained on a natively multimodal dataset such as UBFC-Phys (which contains synchronized facial video, BVP, and EDA) or the SWELL knowledge work dataset, ensuring mutual information exists between the modalities (I(X 
vision
​
 ;Y)>0).   

Confounder Injection on True Pairs: The synthetic scar must be injected directly onto the native facial videos (e.g., the UBFC-Phys video frames) at the strictly defined bias regimes (ρ=0.85,0.5,0.15). This experimental design forces the Stiefel-constrained causal layer to perform its actual intended function: disentangling the synthetic, spurious shortcut (the scar) from the true visual physiological signals (e.g., rPPG chromatic changes, eye blinks, stress-induced micro-expressions).   

Rigorous Baseline Establishment: Before the EQARNB network is retrained on the new dataset, the fundamental baselines must be established. The K1 (Physiology-only) and K1b (Standard ERM Multimodal) kill-switch baselines must be trained across all three bias regimes using LOSO cross-validation. This will establish the performance ceiling of the physiological modality and provide a quantifiable metric for the extent of shortcut learning occurring in standard architectures.   

Section 5: Immediate Adversarial Directives
The local Antigravity CLI Agent must execute the following immediate technical tasks to begin dismantling the technical debt in the repository. The human developer is instructed to observe the agent's execution, verify the mathematical outputs, and ensure that no legacy checkpoints are utilized during this phase.

[ACTION FOR LOCAL AGENT]: Write a python script using cv2 and mediapipe to re-derive the inter-ocular distance (IOD) across all frames of subject 1, clip 1, and calculate the mean ratio to prove or disprove the 3.566283 constant. Target MediaPipe FaceLandmarker indices 33 and 263 (outer eye corners) as well as 468 and 473 (pupils) to locate the exact spatial anchor previously used. Output the raw float and output the standard deviation of the ratio across the temporal axis to ensure spatial stability.

[ACTION FOR LOCAL AGENT]: Write a python script to instantiate the UBFC-Phys data loader and scale the current POS, CHROM, and PBV signal extraction pipeline from N=4 to the full N=56 subjects. The script must iterate over all 56 subjects, apply the H.264 compression artifact utilizing FFmpeg at Constant Rate Factor (CRF) 28 and CRF 36, and output a CSV containing the raw Signal-to-Noise Ratio (SNR) and Mean Absolute Error (MAE) for the recovered rPPG waveform against the ground-truth Empatica E4 wrist BVP. Crucially, this execution must strictly bypass the EMA smoothing function ($\alpha=1.0$) to expose the unmasked signal degradation.

[ACTION FOR LOCAL AGENT]: Write a python script to construct and evaluate the K1 (Physiology-only) baseline kill-switch. Load the WESAD ECG/EDA physiological features (specifically extracting the Root Mean Square of Successive Differences (RMSSD) and Low-Frequency to High-Frequency (LF/HF) power ratio arrays using NeuroKit2). Bypass the vision branch entirely, and train a standard Multilayer Perceptron (MLP) mapping these physiological features directly to the acute stress phase label using Leave-One-Subject-Out (LOSO) cross-validation across the 15 subjects. Output the baseline validation accuracy, AUC, and F1-score to establish the absolute performance ceiling of the physiological modality before EQARNB fusion is attempted.


arxiv.org
A Comprehensive Survey of Bias and Fairness in Speech AI - arXiv
Opens in a new window

pmc.ncbi.nlm.nih.gov
Algorithm fairness in artificial intelligence for medicine and healthcare
Opens in a new window

arxiv.org
efficient riemannian optimization on the stiefel manifold via ... - arXiv
Opens in a new window

aclanthology.org
Riemannian Optimization for LoRA on the Stiefel Manifold
Opens in a new window

arxiv.org
Stiefel Attention: When the Geometry of Transformer Projection
Opens in a new window

alphaxiv.org
Efficient Riemannian Optimization on the Stiefel Manifold ... - alphaXiv
Opens in a new window

researchgate.net
(PDF) Algorithmic Principles of Remote PPG - ResearchGate
Opens in a new window

researchgate.net
(PDF) Enhancing Stress Detection: A Comprehensive Approach
Opens in a new window

intelliprove.com
Tensor term decomposition for remote photoplethysmography
Opens in a new window

pmc.ncbi.nlm.nih.gov
Evaluating Visual Photoplethysmography Method - PMC
Opens in a new window

scispace.com
Camera Measurement of Physiological Vital Signs - SciSpace
Opens in a new window

pmc.ncbi.nlm.nih.gov
Roadmap of remote photoplethysmography from heart rate ... - PMC
Opens in a new window

microsoft.com
The Impact of Video Compression on Remote Cardiac Pulse
Opens in a new window

cmp.felk.cvut.cz
Non-Contact Reflectance Photoplethysmography - CMP
Opens in a new window

pmc.ncbi.nlm.nih.gov
Missed isochromatic cardiac pulsation in remote ... - PMC - NIH
Opens in a new window

computer.org
Multimodal Deep Learning for Remote Stress Estimation Using CCT
Opens in a new window

pmc.ncbi.nlm.nih.gov
Enhancing Stress Detection: A Comprehensive Approach through
Opens in a new window

researchgate.net
(PDF) UBFC-Phys: A Multimodal Database For Psychophysiological
Opens in a new window

computer.org
The ForDigitStress Dataset: A Multi-Modal Dataset for Automatic
Opens in a new window

sorrento21.github.io
Camera-Based Remote Physiology Sensing for Hundreds of
Opens in a new window

arxiv.org
Electrodermal Activity (EDA) for Stress Detection - arXiv
Opens in a new window

openreview.net
rPPG-Toolbox: Deep Remote PPG Toolbox - OpenReview
Opens in a new window

developers.google.com
Face landmark detection guide | Google AI Edge
Opens in a new window

github.com
mediapipe/docs/solutions/face_mesh.md at master - GitHub
Opens in a new window

selvamsubbiah.com
Face Landmarks Detection using MediaPipe Face Mesh in
Opens in a new window

canthion.com
Methodology - Canthion
Opens in a new window

scitepress.org
Face2Cam: A User-to-Webcam Distance Estimation Image Dataset
Opens in a new window

openaccess.thecvf.com
Multimodal Deep Learning for Remote Stress Estimation Using CCT
Opens in a new window

pmc.ncbi.nlm.nih.gov
Strategies for Reliable Stress Recognition: A Machine Learning
Opens in a new window

arxiv.org
Privacy-Preserving On-Device Emotion Recognition from Biosignals
Opens in a new window

dokumen.pub
Shaping AI: Ethics, Society, and the Future of Technology
Opens in a new window

arxiv.org
Benchmarking Fairness in Multimodal LLM Unlearning - arXiv
Opens in a new window

huggingface.co
Daily Papers - Hugging Face
Opens in a new window

researchgate.net
Stress detection with encoding physiological signals and
Opens in a new window

pmc.ncbi.nlm.nih.gov
Global Stress Detection Framework Combining a Reduced Set of
Opens in a new window

arxiv.org
Ensemble Machine Learning Model Trained on a New Synthesized
Opens in a new window

repository.iiitd.edu.in
Towards Human-Centered Data-Driven Emotion ... - IIITD Repository
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window




Based on the provided defense and notebook sources, here is a recheck and synthesis of the critical technical areas to fortify your R&D roadmap and defense preparations:

**1. rPPG Extraction and Video Compression (Line A)**
The literature confirms that video compression algorithms like H.264 heavily impact imaging photoplethysmography (iPPG) because they use chroma subsampling and block-based transformations (like the Discrete Cosine Transform) to discard subtle pixel variations where the blood volume pulse resides. Furthermore, standard rPPG algorithms like CHROM and POS, while mathematically designed to mitigate motion by projecting RGB signals into orthogonal chrominance vectors to remove specular reflection, still conflate isochromatic components with the true chromatic signal. This validates the adversarial requirement to test the raw Signal-to-Noise Ratio (SNR) degradation without the masking effect of heavy temporal smoothing.

**2. MediaPipe Landmarks and IOD Re-derivation**
To fix the chain of custody break regarding the facial cropping rule, the spatial anchor points must be dynamically derived. The MediaPipe Face Landmarker outputs exactly 478 3D landmarks. To calculate the Inter-Ocular Distance (IOD), the system should target the specific indices for the pupils (indices 468 and 473) or the outer eye corners (indices 33 and 263). Dynamic calculation against the population-mean IOD of approximately $62$ mm is required to account for variations in camera distance and focal length.

**3. Stiefel Manifold Optimization (Line B)**
The theoretical defense of the EQARNB causal layer is mathematically sound if optimized correctly. Enforcing orthogonality constraints on parameter matrices stabilizes training dynamics and enriches feature representation. Because standard exponential mapping on the Stiefel manifold requires computationally prohibitive matrix inversion, utilizing an iterative Cayley transform allows the network to maintain strict orthonormality constraints through efficient matrix multiplications without massive computational overhead.

**4. Benchmarking Rigor and Cross-Validation**
For the data generation redesign, the sources emphasize strict evaluation protocols. The UBFC-Phys dataset contains $N=56$ subjects subjected to standardized tasks designed to induce varying stress levels, making a sample size of $N=4$ fundamentally insufficient. For physiological baselines utilizing datasets like WESAD, extracting standard Heart Rate Variability (HRV) features such as RMSSD and the LF/HF power ratio from extracted pulse intervals is standard practice. Across both datasets, the literature mandates strict Leave-One-Subject-Out (LOSO) cross-validation to eliminate information leakage and properly account for inter-subject variability in baseline hemodynamics.