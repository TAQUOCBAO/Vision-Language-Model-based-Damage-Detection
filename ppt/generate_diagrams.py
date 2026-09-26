#!/usr/bin/env python3
"""Generate high-resolution architecture and metrics diagrams for PPT presentation."""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

os.makedirs('ppt/figures', exist_ok=True)

# Set global styles
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

def create_qwen3_vl_architecture_diagram():
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.5)
    ax.axis('off')
    
    # Background canvas
    fig.patch.set_facecolor('#0B132B')
    ax.set_facecolor('#0B132B')
    
    # Title & Subtitle
    ax.text(7.5, 8.0, "Qwen3-VL-8B-Instruct Architecture & LoRA Adaptation Strategy", 
            ha='center', va='center', color='#F8FAFC', fontsize=18, fontweight='bold')
    ax.text(7.5, 7.6, "Specialized Multimodal Adaptation for Civil Structural Damage Diagnosis", 
            ha='center', va='center', color='#94A3B8', fontsize=12)

    # -------------------------------------------------------------
    # 1. INPUT STAGE (Left)
    # -------------------------------------------------------------
    # Image Input Box
    img_box = patches.FancyBboxPatch((0.5, 4.3), 2.6, 2.6, boxstyle="round,pad=0.1", 
                                     fc='#1E293B', ec='#38BDF8', lw=2)
    ax.add_patch(img_box)
    ax.text(1.8, 6.5, "Inspection Image Input", ha='center', va='center', color='#38BDF8', fontsize=11, fontweight='bold')
    ax.text(1.8, 5.8, "Concrete / Bridge Photo\nVariable Aspect Ratio\n(Max 524,288 Pixels)", 
            ha='center', va='center', color='#E2E8F0', fontsize=9.5)
    ax.text(1.8, 4.8, "[ Dynamic 2D Patches ]\nNaViT-style Grid Split", 
            ha='center', va='center', color='#94A3B8', fontsize=9, style='italic')

    # Text Prompt Input Box
    txt_box = patches.FancyBboxPatch((0.5, 1.0), 2.6, 2.6, boxstyle="round,pad=0.1", 
                                     fc='#1E293B', ec='#A855F7', lw=2)
    ax.add_patch(txt_box)
    ax.text(1.8, 3.2, "Prompt & Task Input", ha='center', va='center', color='#A855F7', fontsize=11, fontweight='bold')
    ax.text(1.8, 2.3, "<image> + System Role\n10 Closed Taxonomy Labels\nimage_id Identifier", 
            ha='center', va='center', color='#E2E8F0', fontsize=9.5)
    ax.text(1.8, 1.4, "Text Tokenizer (BPE)\n151,643 Vocab Size", 
            ha='center', va='center', color='#94A3B8', fontsize=9, style='italic')

    # -------------------------------------------------------------
    # 2. VISION PROCESSING (Top Middle)
    # -------------------------------------------------------------
    # Vision Transformer (ViT)
    vit_box = patches.FancyBboxPatch((3.7, 4.3), 2.8, 2.6, boxstyle="round,pad=0.1", 
                                     fc='#1C2541', ec='#EF4444', lw=2)
    ax.add_patch(vit_box)
    ax.text(5.1, 6.5, "Vision Tower (ViT)", ha='center', va='center', color='#EF4444', fontsize=12, fontweight='bold')
    ax.text(5.1, 5.8, "Native Dynamic Resolution\n2D Rotary Pos Embedding\n32 Transformer Layers", 
            ha='center', va='center', color='#E2E8F0', fontsize=9.5)
    
    # Frozen Badge
    f1 = patches.FancyBboxPatch((4.0, 4.5), 2.2, 0.45, boxstyle="round,pad=0.05", fc='#991B1B', ec='none')
    ax.add_patch(f1)
    ax.text(5.1, 4.72, "FROZEN (0 Train Params)", ha='center', va='center', color='#FEF2F2', fontsize=8.5, fontweight='bold')

    # Multi-Modal Projector / Spatial Merger
    proj_box = patches.FancyBboxPatch((7.0, 4.3), 2.2, 2.6, boxstyle="round,pad=0.1", 
                                      fc='#1C2541', ec='#EF4444', lw=2)
    ax.add_patch(proj_box)
    ax.text(8.1, 6.5, "Spatial Merger", ha='center', va='center', color='#EF4444', fontsize=11, fontweight='bold')
    ax.text(8.1, 5.7, "Compresses 2x2\nVision Tokens -> 1\nLinear Projection to\nLLM Hidden Dim (4096)", 
            ha='center', va='center', color='#E2E8F0', fontsize=9)
    
    f2 = patches.FancyBboxPatch((7.2, 4.5), 1.8, 0.45, boxstyle="round,pad=0.05", fc='#991B1B', ec='none')
    ax.add_patch(f2)
    ax.text(8.1, 4.72, "FROZEN", ha='center', va='center', color='#FEF2F2', fontsize=8.5, fontweight='bold')

    # -------------------------------------------------------------
    # 3. LANGUAGE MODEL BACKBONE & LoRA (Bottom Middle)
    # -------------------------------------------------------------
    llm_box = patches.FancyBboxPatch((3.7, 0.8), 5.5, 2.8, boxstyle="round,pad=0.15", 
                                     fc='#1E293B', ec='#34D399', lw=2.5)
    ax.add_patch(llm_box)
    ax.text(6.45, 3.2, "Qwen3 Language Model Backbone (8B Parameters)", 
            ha='center', va='center', color='#34D399', fontsize=13, fontweight='bold')
    ax.text(6.45, 2.7, "36 Transformer Layers · GQA (Grouped Query Attention) · SwiGLU · RMSNorm", 
            ha='center', va='center', color='#CBD5E1', fontsize=9.5)

    # LoRA Adapter Sub-box
    lora_box = patches.FancyBboxPatch((4.0, 1.05), 4.9, 1.35, boxstyle="round,pad=0.08", 
                                      fc='#064E3B', ec='#10B981', lw=1.8, ls='--')
    ax.add_patch(lora_box)
    ax.text(6.45, 2.1, "LoRA Fine-Tuning Adapters (r=64, alpha=128, BF16)", 
            ha='center', va='center', color='#A7F3D0', fontsize=10.5, fontweight='bold')
    ax.text(6.45, 1.6, "Injected into: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj\nWeight Update: W = W_0 + (alpha / r) * B * A  [~160M Trainable Params]", 
            ha='center', va='center', color='#ECFDF5', fontsize=9)
    ax.text(6.45, 1.2, "TRAINABLE ADAPTERS (Trained via LLaMA-Factory)", 
            ha='center', va='center', color='#34D399', fontsize=8.5, fontweight='bold')

    # -------------------------------------------------------------
    # 4. OUTPUT STAGE (Right)
    # -------------------------------------------------------------
    out_box = patches.FancyBboxPatch((9.8, 1.5), 4.7, 5.4, boxstyle="round,pad=0.15", 
                                     fc='#0F172A', ec='#F59E0B', lw=2.5)
    ax.add_patch(out_box)
    ax.text(12.15, 6.5, "Diagnostic Output & Alignment", ha='center', va='center', color='#F59E0B', fontsize=13, fontweight='bold')
    
    json_text = (
        "Autoregressive Generation -> Deterministic Postprocess:\n\n"
        "{\n"
        '  "image_id": "00002",\n'
        '  "damage_categories": [\n'
        '    "cracks",\n'
        '    "spalling"\n'
        '  ],\n'
        '  "description": "Longitudinal fracture running\n'
        '    across the reinforced concrete beam\n'
        '    with adjacent surface spalling..."\n'
        "}\n\n"
        "[ Post-process: JSON Repair + Synonym Map + Text Sync ]"
    )
    ax.text(12.15, 4.0, json_text, ha='center', va='center', color='#E2E8F0', fontsize=9.5, 
            fontfamily='monospace', bbox=dict(boxstyle='round,pad=0.3', fc='#1E293B', ec='#334155'))

    # -------------------------------------------------------------
    # ARROWS & CONNECTIONS
    # -------------------------------------------------------------
    arrow_style = dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=2.2, color='#38BDF8')
    arrow_purple = dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=2.2, color='#A855F7')
    arrow_gold = dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=2.2, color='#F59E0B')

    # Img -> ViT
    ax.annotate("", xy=(3.7, 5.6), xytext=(3.1, 5.6), arrowprops=arrow_style)
    # ViT -> Projector
    ax.annotate("", xy=(7.0, 5.6), xytext=(6.5, 5.6), arrowprops=arrow_style)
    # Projector -> LLM (Downward curved)
    ax.annotate("", xy=(6.5, 3.6), xytext=(8.1, 4.3),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=2.2, color='#38BDF8',
                                connectionstyle="arc3,rad=-0.3"))
    # Text -> LLM
    ax.annotate("", xy=(3.7, 2.2), xytext=(3.1, 2.2), arrowprops=arrow_purple)
    # LLM -> Output
    ax.annotate("", xy=(9.8, 2.5), xytext=(9.2, 2.5), arrowprops=arrow_gold)

    # Key Takeaways Box (Bottom)
    summary_box = patches.FancyBboxPatch((0.5, 0.1), 14.0, 0.55, boxstyle="round,pad=0.05", fc='#1E293B', ec='#475569')
    ax.add_patch(summary_box)
    ax.text(7.5, 0.37, "Key Advantage: Freezing Vision Tower & Projector preserves general visual acuity and saves ~18GB VRAM; LLM LoRA concentrates adaptation on structural forensic JSON reasoning.", 
            ha='center', va='center', color='#38BDF8', fontsize=9.5, fontweight='bold')

    plt.tight_layout()
    plt.savefig('ppt/figures/qwen3_vl_architecture.png', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print("Saved ppt/figures/qwen3_vl_architecture.png")

def create_bakeoff_metrics_chart():
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    fig.patch.set_facecolor('#0F172A')
    ax.set_facecolor('#1E293B')
    
    tiers = ['Tier A (Baseline SFT)', 'Tier B (Aug + Normalize)', 'Tier C (Hybrid Expansion)']
    f1 = [0.9528, 0.9495, 0.9378]
    meteor = [0.6379, 0.6427, 0.6136]
    weighted = [0.82684, 0.82681, 0.80814]

    x = np.arange(len(tiers))
    width = 0.25

    rects1 = ax.bar(x - width, f1, width, label='Multilabel F1', color='#34D399', edgecolor='#059669', lw=1.5)
    rects2 = ax.bar(x, meteor, width, label='METEOR Score', color='#38BDF8', edgecolor='#0284C7', lw=1.5)
    rects3 = ax.bar(x + width, weighted, width, label='Weighted Score (0.6*F1 + 0.4*MET)', color='#FBBF24', edgecolor='#D97706', lw=1.5)

    ax.set_ylabel('Metric Score (0.0 - 1.0)', color='#F8FAFC', fontsize=12, fontweight='bold')
    ax.set_title('Holdout Validation Performance Across 3 Data Tiers (N=237)', color='#F8FAFC', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers, color='#E2E8F0', fontsize=11, fontweight='bold')
    ax.tick_params(colors='#94A3B8')
    ax.set_ylim(0.5, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#64748B')
    
    # Legend
    legend = ax.legend(loc='upper right', facecolor='#0F172A', edgecolor='#475569', labelcolor='#F8FAFC')
    
    # Annotations
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', color='#F8FAFC', fontsize=9, fontweight='bold')

    # Winner badge on Tier A
    ax.annotate('🏆 WINNER', xy=(0 - width, 0.9528), xytext=(0, 25),
                textcoords="offset points", ha='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='#10B981', ec='none'),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#34D399'),
                color='#FFFFFF', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('ppt/figures/bakeoff_comparison.png', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print("Saved ppt/figures/bakeoff_comparison.png")

def create_vram_comparison_chart():
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
    fig.patch.set_facecolor('#0F172A')
    ax.set_facecolor('#1E293B')

    modes = ['Full Precision\nBF16', '8-Bit\nQuantized', '4-Bit NF4\n(bitsandbytes)']
    vram = [18.2, 10.5, 5.8]
    colors = ['#EF4444', '#FBBF24', '#34D399']

    bars = ax.barh(modes, vram, color=colors, height=0.5, edgecolor='#334155', lw=1.5)
    ax.set_xlabel('Inference VRAM Allocation (GB)', color='#F8FAFC', fontsize=12, fontweight='bold')
    ax.set_title('Inference VRAM Footprint for Qwen3-VL-8B + LoRA Adapter', color='#F8FAFC', fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(colors='#E2E8F0', labelsize=11)
    ax.set_xlim(0, 24)
    ax.grid(axis='x', linestyle='--', alpha=0.3, color='#64748B')

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.4, bar.get_y() + bar.get_height()/2, f'{width:.1f} GB', 
                va='center', color='#F8FAFC', fontsize=11, fontweight='bold')

    ax.axvline(6.0, color='#38BDF8', linestyle=':', lw=2, label='Consumer GPU Limit (RTX 3060 6GB)')
    ax.axvline(16.0, color='#A855F7', linestyle=':', lw=2, label='Mid-tier GPU Limit (RTX 4080 16GB)')
    ax.legend(loc='lower right', facecolor='#0F172A', edgecolor='#475569', labelcolor='#F8FAFC')

    plt.tight_layout()
    plt.savefig('ppt/figures/vram_comparison.png', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print("Saved ppt/figures/vram_comparison.png")

if __name__ == '__main__':
    create_qwen3_vl_architecture_diagram()
    create_bakeoff_metrics_chart()
    create_vram_comparison_chart()
