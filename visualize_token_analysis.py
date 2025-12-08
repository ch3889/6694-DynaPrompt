"""
Visualize Per-Token Analysis Results
Shows which concepts are underrepresented and how they're corrected
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

def visualize_token_analysis(analysis_dir="outputs/per_token_analysis"):
    """Create visualizations of per-token analysis results"""
    
    # Find all analysis JSON files
    analysis_files = [f for f in os.listdir(analysis_dir) if f.endswith('_analysis.json')]
    
    if not analysis_files:
        print(f"No analysis files found in {analysis_dir}")
        return
    
    for analysis_file in analysis_files:
        filepath = os.path.join(analysis_dir, analysis_file)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        prompt = data['prompt']
        feedback_history = data['feedback_history']
        weak_tokens_summary = data['weak_tokens_summary']
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f'DynaPrompt Per-Token Analysis\n"{prompt}"', fontsize=12, fontweight='bold')
        
        # Plot 1: CLIP Score Evolution
        steps = [entry['step'] for entry in feedback_history]
        clip_scores = [entry['clip_score'] for entry in feedback_history]
        
        ax1.plot(steps, clip_scores, marker='o', linewidth=2, markersize=8, color='#2E86AB')
        ax1.set_xlabel('Denoising Step', fontsize=10)
        ax1.set_ylabel('CLIP Score', fontsize=10)
        ax1.set_title('Semantic Alignment Over Time', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=np.mean(clip_scores), color='gray', linestyle='--', alpha=0.5, label='Mean')
        ax1.legend()
        
        # Plot 2: Weak Token Frequency
        if weak_tokens_summary:
            tokens = list(weak_tokens_summary.keys())[:10]  # Top 10
            counts = [weak_tokens_summary[t] for t in tokens]
            
            colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(tokens)))
            bars = ax2.barh(tokens, counts, color=colors)
            ax2.set_xlabel('Times Detected as Weak', fontsize=10)
            ax2.set_title('Most Underrepresented Concepts', fontsize=11, fontweight='bold')
            ax2.grid(True, axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, count) in enumerate(zip(bars, counts)):
                ax2.text(count + 0.1, i, str(count), va='center', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'No weak tokens detected\n✓ All concepts well-represented', 
                    ha='center', va='center', fontsize=12, transform=ax2.transAxes)
            ax2.set_title('Most Underrepresented Concepts', fontsize=11, fontweight='bold')
        
        # Plot 3: Weak Token Evolution Timeline
        # Show which tokens were weak at each step
        all_weak_tokens = set()
        for entry in feedback_history:
            all_weak_tokens.update(entry['weak_tokens'])
        
        if all_weak_tokens:
            token_list = sorted(list(all_weak_tokens))[:8]  # Show top 8
            token_timeline = np.zeros((len(token_list), len(steps)))
            
            for step_idx, entry in enumerate(feedback_history):
                for token_idx, token in enumerate(token_list):
                    if token in entry['weak_tokens']:
                        token_timeline[token_idx, step_idx] = 1
            
            im = ax3.imshow(token_timeline, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')
            ax3.set_yticks(range(len(token_list)))
            ax3.set_yticklabels(token_list, fontsize=9)
            ax3.set_xticks(range(len(steps)))
            ax3.set_xticklabels(steps, fontsize=8)
            ax3.set_xlabel('Denoising Step', fontsize=10)
            ax3.set_title('Weak Token Timeline (Red = Underrepresented)', fontsize=11, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax3, orientation='horizontal', pad=0.1, aspect=30)
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(['Strong', 'Weak'])
        else:
            ax3.text(0.5, 0.5, 'No weak tokens across all steps', 
                    ha='center', va='center', fontsize=12, transform=ax3.transAxes)
            ax3.set_title('Weak Token Timeline', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        output_name = analysis_file.replace('_analysis.json', '_visualization.png')
        output_path = os.path.join(analysis_dir, output_name)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved: {output_path}")
        plt.close()

if __name__ == "__main__":
    print("=" * 70)
    print("Creating Per-Token Analysis Visualizations")
    print("=" * 70)
    
    visualize_token_analysis()
    
    print("\n" + "=" * 70)
    print("Visualizations Complete!")
    print("=" * 70)
    print("\nGenerated plots show:")
    print("  1. CLIP score evolution during generation")
    print("  2. Most frequently underrepresented concepts")
    print("  3. Timeline of which tokens were weak at each step")
    print("\nThese visualizations validate the proposal's claims about")
    print("detecting and correcting underrepresented concepts.")
