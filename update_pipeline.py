import re

filepath = r'scripts\ubfc_leakage\run_full_pos_pipeline.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the max_cross_corr definition
old_func = re.search(r'def max_cross_corr\(x, y, max_lag_samples\):.*?return 0\.0\n', content, re.DOTALL)
if old_func:
    new_func = '''def point_to_point_corr(x, y):
    if len(x) > 1 and np.std(x) > 1e-6 and np.std(y) > 1e-6:
        r, _ = stats.pearsonr(x, y)
        return r if not np.isnan(r) else 0.0
    return 0.0
'''
    content = content.replace(old_func.group(0), new_func)

# Replace the calls in main
content = content.replace('best_r = max_cross_corr(clip[\'rppg_filt\'], clip[\'bvp_filt\'], max_lag_samples)', 
                          'best_r = point_to_point_corr(clip[\'rppg_filt\'], clip[\'bvp_filt\'])')
content = content.replace('best_r_null = max_cross_corr(c1[\'rppg_filt\'], c2[\'bvp_filt\'], max_lag_samples)', 
                          'best_r_null = point_to_point_corr(c1[\'rppg_filt\'], c2[\'bvp_filt\'])')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated run_full_pos_pipeline.py successfully.")
