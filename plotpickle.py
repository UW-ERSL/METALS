import pickle
import matplotlib.pyplot as plt

# Load the pickle file
with open('C:\LSR_METALS_SAVED\compliance_vs_criticality_EdgeCantilever_Temp.pkl', 'rb') as f:
    data = pickle.load(f)

thresholds = data['thresholds']
final_compliances = data['final_compliances']

# Remove the entry for criticality 1.25
target_criticality = 100

if target_criticality in thresholds:
    idx = thresholds.index(target_criticality)
    thresholds.pop(idx)
    final_compliances.pop(idx)
    print(f"Removed entry for criticality {target_criticality}")
else:
    print(f"Criticality {target_criticality} not found in thresholds.")
# thresholds.append(1.25)
# final_compliances.append(3.02445)
# Sort both lists by thresholds in ascending order
sorted_pairs = sorted(zip(thresholds, final_compliances))
thresholds, final_compliances = zip(*sorted_pairs)
thresholds = list(thresholds)
final_compliances = list(final_compliances)
with open('compliance_vs_criticality_EdgeCantilever_Reduced_8patches_correct_random.pkl', 'wb') as f:
    pickle.dump({'thresholds': thresholds, 'final_compliances': final_compliances}, f)


# Plot
plt.figure()
plt.plot(thresholds, final_compliances, marker='o')
plt.xlabel('Criticality Index Threshold', fontsize=16)
plt.ylabel('Final Compliance', fontsize=16)
plt.title('Final Compliance vs. Criticality Index Threshold', fontsize=18)
plt.gca().invert_xaxis()
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.show()
# ...existing code...