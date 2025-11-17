import matplotlib.pyplot as plt
import re

def parse_log(filename):
    """Parse log file and extract frames and mean returns"""
    frames = []
    returns = []
    
    with open(filename, 'r') as f:
        for line in f:
            # Look for lines starting with "U " (update lines)
            if line.startswith('U '):
                # Extract frame number
                frame_match = re.search(r'F (\d+)', line)
                # Extract mean return (first number after rR:mean/std/min/max)
                return_match = re.search(r'rR:mean/std/min/max ([\d.]+)', line)
                
                if frame_match and return_match:
                    frames.append(int(frame_match.group(1)))
                    returns.append(float(return_match.group(1)))
    
    return frames, returns

baseline_v2_frames, baseline_v2_returns = parse_log('storage/redblue_3stage_baseline_1M_v2/log_baseline_1M_v2.txt')
baseline_fail_frames, baseline_fail_returns = parse_log('storage/redblue_3stage_baseline_1M/log_3stage_fail.txt')
cb_frames, cb_returns = parse_log('storage/redblue_3stage_cb_1M/log_cb.txt')

# Create figure
plt.figure(figsize=(10, 6))

# Plot 3 curves
plt.plot([f/1e6 for f in baseline_v2_frames], baseline_v2_returns, 
         label='Baseline (partial forgetting)', linewidth=2, alpha=0.8)
plt.plot([f/1e6 for f in baseline_fail_frames], baseline_fail_returns, 
         label='Baseline (catastrophic collapse)', linewidth=2, alpha=0.8)
plt.plot([f/1e6 for f in cb_frames], cb_returns, 
         label='CB (stable)', linewidth=2, alpha=0.8)

# Add vertical lines for stage transitions
plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='6x6→7x7 transition')
plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='7x7→8x8 transition')

# Highlight collapse region
plt.axvspan(1.8, 2.0, alpha=0.2, color='red', label='Collapse region')

# Labels and formatting
plt.xlabel('Training Frames (Millions)', fontsize=12)
plt.ylabel('Mean Success Rate', fontsize=12)
plt.title('RedBlueDoors Training: Baseline Catastrophic Forgetting vs CB Stability', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim(-0.05, 1.05)

# Save figure
plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
plt.savefig('training_curves.pdf', bbox_inches='tight')
print("Training curves saved to /mnt/user-data/outputs/")
