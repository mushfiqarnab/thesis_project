import json

with open('outputs/analysis/thesis_final_comprehensive_report.json', 'r') as f:
    report = json.load(f)

print('='*90)
print('SANITY CHECK - CONFUSION MATRICES vs ACCURACY/PRECISION/RECALL')
print('='*90)

all_consistent = True

for model_name in ['Baseline', 'Counterfactual', 'Fairness-Repaired']:
    cm = report['models'][model_name]['confusion_matrix']
    metrics = report['models'][model_name]['metrics']
    
    TN, FP = cm['TN'], cm['FP']
    FN, TP = cm['FN'], cm['TP']
    
    total = TN + FP + FN + TP
    calculated_acc = (TN + TP) / total
    claimed_acc = metrics['Accuracy']
    
    calculated_prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    claimed_prec = metrics['Precision']
    
    calculated_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    claimed_recall = metrics['Recall']
    
    acc_match = abs(claimed_acc - calculated_acc) < 0.001
    prec_match = abs(claimed_prec - calculated_prec) < 0.001
    recall_match = abs(claimed_recall - calculated_recall) < 0.001
    
    print(f'\n{model_name}:')
    print(f'  Confusion Matrix: TN={TN}, FP={FP}, FN={FN}, TP={TP}')
    print(f'  Total samples: {total}')
    
    acc_status = "MATCH" if acc_match else "MISMATCH"
    print(f'  Accuracy:  Claimed={claimed_acc:.4f}  Calculated={calculated_acc:.4f}  [{acc_status}]')
    
    prec_status = "MATCH" if prec_match else "MISMATCH"
    print(f'  Precision: Claimed={claimed_prec:.4f}  Calculated={calculated_prec:.4f}  [{prec_status}]')
    
    recall_status = "MATCH" if recall_match else "MISMATCH"
    print(f'  Recall:    Claimed={claimed_recall:.4f}  Calculated={calculated_recall:.4f}  [{recall_status}]')
    
    model_consistent = acc_match and prec_match and recall_match
    if model_consistent:
        print(f'  ✅ ALL METRICS CONSISTENT')
    else:
        print(f'  ❌ INCONSISTENCIES FOUND')
        all_consistent = False

print('\n' + '='*90)
if all_consistent:
    print('✅ ALL MODELS HAVE INTERNALLY CONSISTENT METRICS - READY FOR THESIS')
else:
    print('❌ SOME INCONSISTENCIES DETECTED - REVIEW REQUIRED')
print('='*90)
