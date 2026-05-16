import matplotlib.pyplot as plt

# --- DATA ENTRY (Extracted from your Phase 3 Results) ---
# Serialized metric history vectors extracted from the advanced 20-epoch training execution
epochs = list(range(20))

# Cross-Entropy Loss optimization convergence history metrics
train_loss = [0.2021, 0.0855, 0.0583, 0.0596, 0.0369, 0.0391, 0.0294, 0.0378, 0.0330, 0.0250, 0.0325, 0.0240, 0.0254, 0.0214, 0.0225, 0.0194, 0.0234, 0.0195, 0.0278, 0.0193]
val_loss = [0.1772, 0.1278, 0.0868, 0.0807, 0.0875, 0.0594, 0.0870, 0.0710, 0.0560, 0.0532, 0.0702, 0.0592, 0.0524, 0.0752, 0.0604, 0.0779, 0.0588, 0.0725, 0.0746, 0.0578]

# Categorical accuracy progression history metrics
train_acc = [0.8972, 0.9612, 0.9728, 0.9719, 0.9839, 0.9800, 0.9854, 0.9827, 0.9821, 0.9875, 0.9863, 0.9883, 0.9892, 0.9910, 0.9910, 0.9889, 0.9907, 0.9910, 0.9880, 0.9928]
val_acc = [0.9369, 0.9570, 0.9698, 0.9714, 0.9703, 0.9830, 0.9698, 0.9745, 0.9814, 0.9820, 0.9745, 0.9788, 0.9825, 0.9735, 0.9793, 0.9729, 0.9804, 0.9735, 0.9735, 0.9804]

# --- VISUALIZATION SETUP ---
plt.figure(figsize=(14, 6))
plt.suptitle('Clinical Performance Analysis: EfficientNet-B0 Pneumonia Detection', fontsize=16)

# Plot 1: Clinical Cross-Entropy Loss Convergence Trace
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b-o', label='Training Loss', markersize=4)
plt.plot(epochs, val_loss, 'r--x', label='Validation Loss', markersize=4)
plt.title('Clinical Loss (Error Reduction)')
plt.xlabel('Epochs')
plt.ylabel('CrossEntropy Loss')
plt.legend()
plt.grid(True, alpha=0.2)

# Plot 2: Diagnostic Categorical Accuracy Progression Trace
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'g-o', label='Training Accuracy', markersize=4)
plt.plot(epochs, val_acc, 'm--x', label='Validation Accuracy', markersize=4)
plt.title('Diagnostic Accuracy (Correct Classifications)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.ylim(0.85, 1.0) # We zoom in on the 85-100% range to see the fine-tuning.
plt.legend()
plt.grid(True, alpha=0.2)

# --- FINAL OUTPUT ---
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('full_training_performance.png', dpi=300) # Save as high-res for your report.
plt.show()

print("✅ Professional performance curves saved as 'full_training_performance.png'")
