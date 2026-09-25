import os
ckpts = [
    "outputs/checkpoints/counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best_stiefel.pt",
    "outputs/fair_model_best.pth",
    "outputs/lambda_sweep_ckpts/lambda_0.0_best.pth"
]
with open('scratch_eval2.py', 'w') as f:
    f.write(open('scratch_eval.py').read().replace('ckpts = [', 'ckpts = ' + str(ckpts) + ' #'))
