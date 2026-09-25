# Claude's New TODOs from files(3)

- **chapter_3.tex**: The markdown states 1,140,133 trainable parameters in one place and
  "approximately 2.5M parameters" for MobileNetV3-Small in another. Report the full
  model's parameter count and, if part of the backbone is frozen, the trainable count,
  so the two figures are not read as contradictory.


- **chapter_3.tex**: Define "activation energy" precisely (for example, the squared L2 norm
  over channels at each spatial location of A), and state how M is resized to h x w.


- **chapter_3.tex**: Check the code and state (a) which loss trains Focus_auto (for example,
  a regression or distillation loss to the privileged Focus), (b) whether the two are on
  the same scale -- the privileged Focus = log(1 + r) is unbounded, while Focus_auto is a
  sigmoid in (0, 1), so exp(-kappa * Focus) can be weaker at inference than in training
  unless the target is rescaled -- and (c) which focus the gate uses during training.
  Report this in Chapter 3; a train/inference mismatch here would need to be listed as a
  limitation in Chapter 4.


- **chapter_3.tex**: State whether G is a scalar or a 192-dimensional vector. The fusion
  below uses element-wise multiplication, which implies a vector, while exp(-kappa*Focus)
  is a scalar applied to every dimension. The factor $\exp(-\kappa \cdot \text{Focus})$ reduces the gate, and hence the weight given to visual features, when those features concentrate on the scar region.


- **chapter_3.tex**: Give the exact definitions of L_causal-inv, L_latent-inv, L_DP and L_EO
  (distance measure, and which surrogate functions are used for DP and EO).


- **chapter_3.tex**: State the window length and stride, which video frame represents each
  window, and which BVP/EDA features make up X_P (e.g. mean IBI, SDNN, RMSSD, EDA
  statistics).


- **chapter_3.tex**: State how the counterfactual is produced. Because the scar is synthetic,
  the ideal counterfactual is the original frame before the scar was added, which differs
  from the scarred frame only inside the scar; if the code instead removes the scar from
  the scarred frame (e.g. by blurring inside the mask), state that and its limitation.
- **chapter_3.tex**: State whether the train/validation/test split is by participant or by
  window. With eight participants, a window-level split places data from the same
  person in every split and makes test accuracy optimistic.


- **chapter_3.tex**: State the scar-label association in the training set. If training uses
  rho = 0.85, keep the phrase "the association in the training data" below; if training
  is balanced, change the Regime 1 and Regime 3 descriptions accordingly.


- **chapter_3.tex**: Confirm in the code that Model C is named CGF (Causal Gated Fusion),
  not "CGP"; the markdown used both.


- **chapter_2_revisions.tex**: Confirm in Sabour et al. (2023) that UBFC-Phys records BVP and EDA
  with the Empatica E4, and check which BVP/EDA features the model uses (e.g. mean IBI,
  SDNN, RMSSD, EDA statistics) so the second-to-last sentence matches the code.


- **chapter_3.tex**: The markdown states 1,140,133 trainable parameters in one place and
  "approximately 2.5M parameters" for MobileNetV3-Small in another. Report the full
  model's parameter count and, if part of the backbone is frozen, the trainable count,
  so the two figures are not read as contradictory.


- **chapter_3.tex**: Define "activation energy" precisely (for example, the squared L2 norm
  over channels at each spatial location of A), and state how M is resized to h x w.


- **chapter_3.tex**: Check the code and state (a) which loss trains Focus_auto (for example,
  a regression or distillation loss to the privileged Focus), (b) whether the two are on
  the same scale -- the privileged Focus = log(1 + r) is unbounded, while Focus_auto is a
  sigmoid in (0, 1), so exp(-kappa * Focus) can be weaker at inference than in training
  unless the target is rescaled -- and (c) which focus the gate uses during training.
  Report this in Chapter 3; a train/inference mismatch here would need to be listed as a
  limitation in Chapter 4.


- **chapter_3.tex**: State whether G is a scalar or a 192-dimensional vector. The fusion
  below uses element-wise multiplication, which implies a vector, while exp(-kappa*Focus)
  is a scalar applied to every dimension. The factor $\exp(-\kappa \cdot \text{Focus})$ reduces the gate, and hence the weight given to visual features, when those features concentrate on the scar region.


- **chapter_3.tex**: Give the exact definitions of L_causal-inv, L_latent-inv, L_DP and L_EO
  (distance measure, and which surrogate functions are used for DP and EO).


- **chapter_3.tex**: State the window length and stride, which video frame represents each
  window, and which BVP/EDA features make up X_P (e.g. mean IBI, SDNN, RMSSD, EDA
  statistics).


- **chapter_3.tex**: State how the counterfactual is produced. Because the scar is synthetic,
  the ideal counterfactual is the original frame before the scar was added, which differs
  from the scarred frame only inside the scar; if the code instead removes the scar from
  the scarred frame (e.g. by blurring inside the mask), state that and its limitation.
- **chapter_3.tex**: State whether the train/validation/test split is by participant or by
  window. With eight participants, a window-level split places data from the same
  person in every split and makes test accuracy optimistic.


- **chapter_3.tex**: State the scar-label association in the training set. If training uses
  rho = 0.85, keep the phrase "the association in the training data" below; if training
  is balanced, change the Regime 1 and Regime 3 descriptions accordingly.


- **chapter_3.tex**: Confirm in the code that Model C is named CGF (Causal Gated Fusion),
  not "CGP"; the markdown used both.


- **chapter_4.tex**: The markdown attributes the 7.8-point gap "exclusively" to the scar.
  That requires evidence such as Model A's accuracy on scar-removed or inverted
  (rho = 0.15) test data; add that result, or keep the more cautious wording above.


- **chapter_4.tex**: Add the regime x mode x seed table for the final version.


- **chapter_4.tex**: State the CPU model, batch size, number of threads, and whether the
  latency includes preprocessing (face detection, cropping) or only the network.


- **chapter_5.tex**: Add a citation for the ViT-B/16 figures (Dosovitskiy et al., ICLR 2021)
  if you keep this comparison. Do not reuse the markdown's latency table: its
  MobileNetV3-Large figure (~1.01 ms on a mobile CPU) conflicts with the 51 ms on a
  Pixel 1 reported by Howard et al.


- **approval.tex**: The semester and acceptance month below ("Summer 2025", "October 2025")
  conflict with the title page (September 2026). Use your actual semester and the
  expected acceptance month.


