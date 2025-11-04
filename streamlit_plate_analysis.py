#!/usr/bin/env python3
"""
GloSensor Assay Analysis Tool
A Streamlit application for analyzing GloSensor assay data
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import tempfile
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pickle

from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, PageBreak, Image, Spacer
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER

# Vessel protocols lookup table
VESSEL_PROTOCOLS = {
    '96-well': {
        'surface_area': '0.3 cm²',
        'plating_medium': '100 μl',
        'dilution_medium': '2 × 25 μl',
        'dna_amount': '0.2 μg',
        'dna_lipofectamine': '0.5 μl',
        'rna_amount': '5 pmol',
        'rna_lipofectamine': '0.25 μl'
    },
    '24-well': {
        'surface_area': '2 cm²',
        'plating_medium': '500 μl',
        'dilution_medium': '2 × 50 μl',
        'dna_amount': '0.8 μg',
        'dna_lipofectamine': '2.0 μl',
        'rna_amount': '20 pmol',
        'rna_lipofectamine': '1.0 μl'
    },
    '12-well': {
        'surface_area': '4 cm²',
        'plating_medium': '1 ml',
        'dilution_medium': '2 × 100 μl',
        'dna_amount': '1.6 μg',
        'dna_lipofectamine': '4.0 μl',
        'rna_amount': '40 pmol',
        'rna_lipofectamine': '2.0 μl'
    },
    '6-well': {
        'surface_area': '10 cm²',
        'plating_medium': '2 ml',
        'dilution_medium': '2 × 250 μl',
        'dna_amount': '4.0 μg',
        'dna_lipofectamine': '10 μl',
        'rna_amount': '100 pmol',
        'rna_lipofectamine': '5 μl'
    },
    '60-mm': {
        'surface_area': '20 cm²',
        'plating_medium': '5 ml',
        'dilution_medium': '2 × 0.5 ml',
        'dna_amount': '8.0 μg',
        'dna_lipofectamine': '20 μl',
        'rna_amount': '200 pmol',
        'rna_lipofectamine': '10 μl'
    },
    '10-cm': {
        'surface_area': '60 cm²',
        'plating_medium': '15 ml',
        'dilution_medium': '2 × 1.5 ml',
        'dna_amount': '24 μg',
        'dna_lipofectamine': '60 μl',
        'rna_amount': '600 pmol',
        'rna_lipofectamine': '30 μl'
    }
}

def save_analysis_state(export_df, df_pre, df_post, labels_matrix, fc_matrix, meta, excluded_wells):
    state = {
        'export_df': export_df,
        'df_pre': df_pre,
        'df_post': df_post,
        'labels_matrix': labels_matrix,
        'fc_matrix': fc_matrix,
        'meta': meta,
        'excluded_wells': excluded_wells,
        'version': '1.0'
    }
    return pickle.dumps(state)

def load_analysis_state(uploaded_file):
    try:
        state = pickle.loads(uploaded_file.read())
        return state
    except Exception as e:
        st.error(f"Error loading analysis file: {e}")
        return None

def analyze_replicates(export_df, excluded_wells=None):
    if excluded_wells is None:
        excluded_wells = {}
    
    replicate_data = []
    grouped = export_df.groupby('Label')
    
    for label, group in grouped:
        if label and label.strip() != '':
            excluded_for_label = excluded_wells.get(label, [])
            filtered_group = group[~group['Well'].isin(excluded_for_label)]
            
            n_total = len(group)
            n_included = len(filtered_group)
            
            if n_total > 1:
                if n_included > 0:
                    mean_fc = filtered_group['Fold Change'].mean()
                    std_fc = filtered_group['Fold Change'].std() if n_included > 1 else 0
                    cv_percent = (std_fc / mean_fc * 100) if mean_fc != 0 else 0
                    
                    wells_display = []
                    for _, row in group.iterrows():
                        well = row['Well']
                        if well in excluded_for_label:
                            wells_display.append(f"~~{well}~~")
                        else:
                            wells_display.append(well)
                    
                    fcs_display = []
                    for _, row in group.iterrows():
                        well = row['Well']
                        fc = row['Fold Change']
                        if well in excluded_for_label:
                            fcs_display.append(f"~~{fc:.2f}~~")
                        else:
                            fcs_display.append(f"{fc:.2f}")
                    
                    replicate_data.append({
                        'Label': label,
                        'n': f"{n_included}/{n_total}",
                        'Wells': ', '.join(wells_display),
                        'Mean FC': round(mean_fc, 3),
                        'Std Dev': round(std_fc, 3),
                        'CV%': round(cv_percent, 1),
                        'Individual FCs': ', '.join(fcs_display)
                    })
    
    if replicate_data:
        return pd.DataFrame(replicate_data)
    else:
        return None

def process_sheet(xls, sheet_name):
    try:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        if "Time (min)" not in df.columns:
            st.error(f"{sheet_name}: Missing 'Time (min)' column.")
            return None, None
        if len(df) < 3:
            st.warning(f"{sheet_name}: Not enough timepoints (≥3 required).")
            return None, None
        last3_avg = df.iloc[-3:, 1:].mean(axis=0).round(2)
        return df, last3_avg
    except Exception as e:
        st.error(f"Error reading {sheet_name}: {e}")
        return None, None

def create_individual_well_kinetics(df_pre, df_post, labels_matrix, fc_matrix):
    fig, axes = plt.subplots(8, 12, figsize=(24, 14))
    fig.patch.set_facecolor('white')
    
    wells = []
    for row in "ABCDEFGH":
        for col in range(1, 13):
            wells.append(f"{row}{col}")
    
    highlighted_wells = []
    
    for idx, well in enumerate(wells):
        row_idx = idx // 12
        col_idx = idx % 12
        ax = axes[row_idx, col_idx]
        ax.set_facecolor('white')
        
        if well in df_pre.columns:
            fc = fc_matrix[row_idx, col_idx]
            is_highlighted = not pd.isna(fc) and fc > 2.0
            
            if is_highlighted:
                highlighted_wells.append(well)
                ax.set_facecolor('#E8F5E9')
            
            ax.plot(df_pre['Time (min)'], df_pre[well], 
                   'o-', linewidth=1, markersize=3, color='black', alpha=0.7)
            ax.plot(df_post['Time (min)'], df_post[well], 
                   's-', linewidth=2.5, markersize=4, color='black')
            
            label = labels_matrix[row_idx, col_idx]
            title = f"{well}: {label}" if label else well
            
            if not pd.isna(fc):
                title += f"\nFC: {fc:.2f}"
            
            if is_highlighted:
                ax.set_title(title, fontsize=9, fontweight='bold')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#4CAF50')
                    spine.set_linewidth(1.2)
            else:
                ax.set_title(title, fontsize=9, fontweight='bold')
                for spine in ax.spines.values():
                    spine.set_edgecolor('white')
                    spine.set_linewidth(1.2)
            
            if col_idx == 0:
                ax.set_ylabel('RLU', fontsize=8)
            if row_idx == 7:
                ax.set_xlabel('Time', fontsize=8)
            
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3, linewidth=0.5)
        else:
            ax.axis('off')
    
    legend_text = "● — Pre (thin)     ■ — Post (thick)     |     Green border & light fill: >2-fold activity"
    fig.text(0.5, 0.005, legend_text, ha='center', va='bottom',
             fontsize=12, family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                      edgecolor='gray', linewidth=0.5, alpha=0.9))
    
    if highlighted_wells:
        note_text = f"Wells with >2-fold activity ({len(highlighted_wells)}): {', '.join(highlighted_wells)}"
        fig.text(0.5, -0.015, note_text, ha='center', fontsize=10, 
                style='italic', color='#2E7D32')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    fig.savefig(tmp.name, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return tmp.name

def create_fold_change_heatmap(fc_matrix, labels_matrix):
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor('white')
    
    valid_fc = [fc_matrix[i, j] for i in range(8) for j in range(12) if not pd.isna(fc_matrix[i, j])]
    if valid_fc:
        vmin = min(valid_fc)
        vmax = max(valid_fc)
    else:
        vmin, vmax = 0.1, 10.0
    
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    
    im = ax.imshow(fc_matrix, cmap='coolwarm', aspect='auto', norm=norm, interpolation='nearest')
    
    ax.set_xticks(np.arange(12))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels([str(i+1) for i in range(12)], fontsize=13)
    ax.set_yticklabels(list('ABCDEFGH'), fontsize=13)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.03, aspect=35)
    cbar.set_label('Fold Change', rotation=270, labelpad=25, fontsize=14)
    cbar.ax.tick_params(labelsize=11)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor('#CCCCCC')
    
    for i in range(8):
        for j in range(12):
            label = labels_matrix[i, j]
            fc = fc_matrix[i, j]
            if not pd.isna(fc):
                if fc < 0.65 or fc > 1.6:
                    text_color = 'white'
                else:
                    text_color = '#333333'
                
                if label:
                    text_str = f"{label}\n{fc:.2f}"
                    fontsize = 10
                else:
                    text_str = f"{fc:.2f}"
                    fontsize = 11
                
                ax.text(j, i, text_str, ha="center", va="center", 
                       color=text_color, fontsize=fontsize, weight='500',
                       linespacing=1.3)
    
    for i in range(9):
        ax.axhline(i - 0.5, color='#999999', linewidth=1.8, zorder=10)
    for j in range(13):
        ax.axvline(j - 0.5, color='#999999', linewidth=1.8, zorder=10)
    
    ax.set_xlabel('Column', fontsize=15, weight='500', labelpad=12, color='#333333')
    ax.set_ylabel('Row', fontsize=15, weight='500', labelpad=12, color='#333333')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#999999')
    ax.spines['left'].set_color('#999999')
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#6495ED', edgecolor='#4A7AC2', linewidth=1.5, label='Decreased activity (FC < 1)'),
        Patch(facecolor='#F5F5F5', edgecolor='#999999', linewidth=1.5, label='No change (FC ≈ 1)'),
        Patch(facecolor='#FF6B6B', edgecolor='#E85555', linewidth=1.5, label='Increased activity (FC > 1)')
    ]
    
    legend = ax.legend(handles=legend_elements, loc='upper center', 
                      bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=True,
                      fontsize=12, edgecolor='#999999', fancybox=False,
                      framealpha=1, borderpad=1.2, columnspacing=2)
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_facecolor('#FAFAFA')
    
    fig.text(0.5, 0.005, f'Data Range: {vmin:.2f} – {vmax:.2f}', 
            ha='center', fontsize=11, color='#666666', weight='500')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    fig.savefig(tmp.name, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return tmp.name

def generate_pdf_bytes(fig_files_info, labels_matrix, fc_values, export_df, df_pre, df_post, meta):
    from datetime import datetime
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4),
                            leftMargin=18, rightMargin=18, topMargin=18, bottomMargin=18)
    
    doc.title = f"{meta.get('experiment_name', 'GloSensor Experiment')} - Analysis Report"
    doc.author = meta.get('name_input', 'Researcher')
    doc.subject = "GloSensor Assay Analysis"
    doc.creator = "GloSensor Analysis Tool"
    
    elements = []
    
    title_style = ParagraphStyle(name='Title', alignment=TA_CENTER, fontSize=24, 
                                 spaceAfter=20, leading=28, textColor=colors.HexColor("#2C3E50"))
    header_style = ParagraphStyle(name='Header', alignment=TA_CENTER, fontSize=14, 
                                  spaceAfter=6, leading=18)
    section_style = ParagraphStyle(name='Section', alignment=TA_CENTER, fontSize=16, 
                                   spaceAfter=8, leading=18, textColor=colors.HexColor("#34495E"))
    small_style = ParagraphStyle(name='Small', alignment=TA_CENTER, fontSize=10, spaceAfter=6)

    elements.append(Spacer(1, 30))
    elements.append(Paragraph("GLOSENSOR ASSAY ANALYSIS REPORT", title_style))
    elements.append(Spacer(1, 20))
    
    current_date = datetime.now().strftime("%B %d, %Y")
    elements.append(Paragraph(f"<b>Date:</b> {current_date}", header_style))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(f"<b>Experiment:</b> {meta.get('experiment_name', 'N/A')}", header_style))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(f"<b>Investigator:</b> {meta.get('name_input', 'N/A')}", header_style))
    elements.append(Spacer(1, 30))
    elements.append(PageBreak())
    
    elements.append(Paragraph("<b>METHODS</b>", section_style))
    elements.append(Spacer(1, 15))
    
    methods_paragraph_style = ParagraphStyle(
        name='Methods',
        fontSize=11,
        leading=16,
        spaceAfter=12,
        alignment=0,
        leftIndent=30,
        rightIndent=30
    )
    
    cells_val = meta.get('cells_seeded', 'N/A')
    media_vol = meta.get('media_volume', '')
    hrs_trans = meta.get('hrs_to_transfection', 'N/A')
    bio_amt = meta.get('biosensor_amount', 'N/A')
    gpcr_amt = meta.get('gpcr_amount', 'N/A')
    hrs_assay = meta.get('hrs_to_assay', 'N/A')
    assay_method = meta.get('assay_method', '')
    transfection_direction = meta.get('transfection_direction', '')
    transfection_agent = meta.get('transfection_agent', '')
    dna_amount_ratio = meta.get('dna_amount_ratio', '')
    agent_amount_ratio = meta.get('agent_amount_ratio', '')
    vessel_type = meta.get('vessel_type', '')
    hrs_to_transfer = meta.get('hrs_to_transfer', 'N/A')
    cells_transferred = meta.get('cells_transferred', 'N/A')
    incubation_time = meta.get('incubation_time', 'N/A')
    serum_free_medium = meta.get('serum_free_medium', 'N/A')
    dna_reagent_incubation = meta.get('dna_reagent_incubation', 'N/A')
    
    transfection_section = f"""
    <b>Experimental Timeline and Transfection Details:</b><br/>
    """
    
    # Add assay method
    if assay_method:
        transfection_section += f"Assay Method: {assay_method}<br/>"
    
    # Add transfection details
    if transfection_direction:
        transfection_section += f"Transfection Direction: {transfection_direction}<br/>"
    
    if transfection_agent:
        transfection_section += f"Transfection Agent: {transfection_agent}<br/>"
    
    transfection_section += "<br/>"
    
    # Add vessel information if Transfer method
    if assay_method == "Transfer method" and vessel_type and vessel_type in VESSEL_PROTOCOLS:
        protocol = VESSEL_PROTOCOLS[vessel_type]
        transfection_section += f"""
    <b>Culture Vessel Information:</b><br/>
    Vessel Type: {vessel_type}<br/>
    Surface Area: {protocol['surface_area']}<br/>
    Plating Medium: {protocol['plating_medium']}<br/>
    Dilution Medium: {protocol['dilution_medium']}<br/>
    <br/>
    """
    
    # Cell seeding
    transfection_section += f"""
    <b>Cell Seeding:</b><br/>
    Number of cells seeded: {cells_val}<br/>
    """
    
    # Timeline to transfection (if forward)
    if transfection_direction == "Forward transfection":
        transfection_section += f"Hours from seeding to transfection: {hrs_trans} hrs<br/>"
    
    transfection_section += "<br/>"
    
    # Transfection details
    transfection_section += f"""
    <b>Transfection Mix Preparation:</b><br/>
    Biosensor DNA: {bio_amt} ng<br/>
    GPCR DNA: {gpcr_amt} ng<br/>
    DNA to Transfection Reagent Ratio: {agent_amount_ratio}<br/>
    Serum-Free Medium: {serum_free_medium} μl<br/>
    DNA + Reagent Incubation Time: {dna_reagent_incubation} min<br/>
    """
    
    if transfection_direction == "Forward transfection" and media_vol:
        transfection_section += f"Cell Culture Medium Volume: {media_vol}<br/>"
    
    transfection_section += "<br/>"
    
    # Method-specific timeline
    if assay_method == "Transfer method":
        transfection_section += f"""
    <b>Post-Transfection Timeline:</b><br/>
    Hours after transfection to transfer: {hrs_to_transfer} hrs<br/>
    Number of cells transferred per well: {cells_transferred}<br/>
    Incubation time after transfer: {incubation_time} hrs<br/>
    """
    else:
        transfection_section += f"""
    <b>Post-Transfection Timeline:</b><br/>
    Hours after transfection to assay: {hrs_assay} hrs<br/>
    """
    
    transfection_section += "<br/>"
    
    elements.append(Paragraph(transfection_section, methods_paragraph_style))
    elements.append(Spacer(1, 10))
    
    methods_content = meta.get('methods_text', '')
    if methods_content and methods_content.strip() != '':
        elements.append(Paragraph("<b>Protocol:</b><br/>" + methods_content.replace('\n', '<br/>'), methods_paragraph_style))
        elements.append(Spacer(1, 10))
    
    notes_content = meta.get('notes_text', '')
    if notes_content and notes_content.strip() != '':
        elements.append(Paragraph("<b>Notes:</b><br/>" + notes_content.replace('\n', '<br/>'), methods_paragraph_style))
    
    elements.append(Spacer(1, 20))
    elements.append(PageBreak())

    for title, filepath in fig_files_info:
        elements.append(Paragraph(f"<b>{title}</b>", section_style))
        elements.append(Spacer(1, 5))
        try:
            elements.append(Image(filepath, width=750, height=450))
        except Exception:
            elements.append(Image(filepath, width=650, height=390))
        elements.append(PageBreak())
        
        if title == "Fold Change Heatmap":
            replicate_df = meta.get('replicate_df')
            if replicate_df is not None and not replicate_df.empty:
                elements.append(Paragraph("<b>REPLICATE ANALYSIS</b>", section_style))
                elements.append(Spacer(1, 10))
                
                rep_header = [['Label', 'n', 'Wells', 'Mean FC', 'Std Dev', 'CV%', 'Individual FCs']]
                rep_data = rep_header
                
                for _, row in replicate_df.iterrows():
                    rep_data.append([
                        str(row['Label']),
                        str(row['n']),
                        str(row['Wells']),
                        f"{row['Mean FC']:.3f}",
                        f"{row['Std Dev']:.3f}",
                        f"{row['CV%']:.1f}",
                        str(row['Individual FCs'])
                    ])
                
                rep_table = Table(rep_data, colWidths=[80, 30, 120, 60, 60, 50, 150])
                
                table_style = [
                    ("GRID", (0,0), (-1,-1), 0.5, colors.black),
                    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#34495E")),
                    ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
                    ("FONTSIZE", (0,0), (-1,-1), 8),
                    ("ALIGN", (0,0), (-1,-1), "CENTER"),
                    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ]
                
                for idx, (_, row) in enumerate(replicate_df.iterrows(), start=1):
                    cv = row['CV%']
                    if cv > 20:
                        color = colors.HexColor("#ffcccc")
                    elif cv > 10:
                        color = colors.HexColor("#fff4cc")
                    else:
                        color = colors.HexColor("#ccffcc")
                    table_style.append(("BACKGROUND", (0,idx), (-1,idx), color))
                
                rep_table.setStyle(TableStyle(table_style))
                
                elements.append(rep_table)
                elements.append(Spacer(1, 10))
                
                legend_text = "Color coding: Green (CV < 10%, Good) | Yellow (CV 10-20%, Moderate) | Red (CV > 20%, High variability)"
                elements.append(Paragraph(legend_text, small_style))
                
                excluded_wells = meta.get('excluded_wells', {})
                total_excluded = sum(len(wells) for wells in excluded_wells.values())
                if total_excluded > 0:
                    excluded_text = f"Note: {total_excluded} well(s) were excluded from replicate calculations (shown with strikethrough)"
                    elements.append(Spacer(1, 5))
                    elements.append(Paragraph(excluded_text, small_style))
                
                elements.append(Spacer(1, 20))
                elements.append(PageBreak())

    elements.append(Paragraph("<b>Raw Data Summary (Pre vs Post by Well)</b>", section_style))
    elements.append(Spacer(1, 10))
    
    wells = [col for col in df_pre.columns if col != 'Time (min)']
    time_points = [f"{t:.1f}" for t in df_pre['Time (min)']]
    
    num_time_cols = len(time_points)
    well_col_width = 50
    time_col_width = 40
    table_width = well_col_width + (num_time_cols * time_col_width)
    
    col_widths = [well_col_width] + [time_col_width] * num_time_cols
    
    wells_per_page = 24
    
    for well_idx in range(0, len(wells), wells_per_page):
        well_subset = wells[well_idx:well_idx+wells_per_page]
        
        pre_header = [['PRE'], ['Well'] + time_points]
        pre_data = pre_header
        for well in well_subset:
            row_letter = well[0]
            col_num = int(well[1:])
            row_idx = ord(row_letter) - ord('A')
            col_idx = col_num - 1
            label = labels_matrix[row_idx, col_idx]
            
            if label:
                well_display = f"{well}: {label}"
            else:
                well_display = well
            
            row = [well_display]
            for time_idx in range(len(df_pre)):
                row.append(f"{df_pre[well].iloc[time_idx]:.1f}")
            pre_data.append(row)
        
        pre_table = Table(pre_data, colWidths=col_widths)
        pre_table.setStyle(TableStyle([
            ("SPAN", (0,0), (-1,0)),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2C5F7E")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,0), 10),
            ("ALIGN", (0,0), (-1,0), "CENTER"),
            ("GRID", (0,1), (-1,-1), 0.5, colors.black),
            ("BACKGROUND", (0,1), (-1,1), colors.HexColor("#4A7BA7")),
            ("TEXTCOLOR", (0,1), (-1,1), colors.whitesmoke),
            ("FONTSIZE", (0,1), (-1,-1), 7),
            ("ALIGN", (0,1), (-1,-1), "CENTER"),
            ("VALIGN", (0,1), (-1,-1), "MIDDLE"),
        ]))
        
        post_header = [['POST'], ['Well'] + time_points]
        post_data = post_header
        for well in well_subset:
            row_letter = well[0]
            col_num = int(well[1:])
            row_idx = ord(row_letter) - ord('A')
            col_idx = col_num - 1
            label = labels_matrix[row_idx, col_idx]
            
            if label:
                well_display = f"{well}: {label}"
            else:
                well_display = well
            
            row = [well_display]
            for time_idx in range(len(df_post)):
                row.append(f"{df_post[well].iloc[time_idx]:.1f}")
            post_data.append(row)
        
        post_table = Table(post_data, colWidths=col_widths)
        post_table.setStyle(TableStyle([
            ("SPAN", (0,0), (-1,0)),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#7E2C2C")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,0), 10),
            ("ALIGN", (0,0), (-1,0), "CENTER"),
            ("GRID", (0,1), (-1,-1), 0.5, colors.black),
            ("BACKGROUND", (0,1), (-1,1), colors.HexColor("#A74A4A")),
            ("TEXTCOLOR", (0,1), (-1,1), colors.whitesmoke),
            ("FONTSIZE", (0,1), (-1,-1), 7),
            ("ALIGN", (0,1), (-1,-1), "CENTER"),
            ("VALIGN", (0,1), (-1,-1), "MIDDLE"),
        ]))
        
        side_by_side = Table([[pre_table, post_table]], colWidths=[table_width, table_width])
        side_by_side.setStyle(TableStyle([
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 5),
            ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ]))
        
        elements.append(side_by_side)
        elements.append(Spacer(1, 20))
        elements.append(PageBreak())
    
    try:
        doc.build(elements)
        return buffer.getvalue()
    except Exception as e:
        st.error(f"Error building PDF: {e}")
        return None

def main():
    st.set_page_config(page_title="GloSensor Assay Analysis", layout="wide", initial_sidebar_state="expanded")
    
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            scroll-behavior: smooth;
        }
        
        /* Main Container Styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }
        
        /* Header Styling */
        .main-header {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            padding: 1.5rem 0 0.5rem 0;
            letter-spacing: -0.02em;
        }
        
        .sub-header {
            font-size: 1.1rem;
            color: #64748b;
            text-align: center;
            padding-bottom: 2rem;
            font-weight: 400;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        }
        
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
            color: #1e293b;
            font-weight: 600;
        }
        
        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: linear-gradient(to right, #f8fafc, #f1f5f9);
            padding: 8px;
            border-radius: 16px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 12px 24px;
            font-size: 0.95rem;
            font-weight: 500;
            border-radius: 12px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            color: #64748b;
            border: none;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e2e8f0;
            color: #475569;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        
        /* Button Styling */
        .stButton > button {
            border-radius: 12px;
            font-weight: 500;
            border: none;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            padding: 0.6rem 1.5rem;
            font-size: 0.95rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            background: white;
            color: #475569;
        }
        
        .stButton > button:hover {
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            transform: translateY(-2px);
        }
        
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        }
        
        /* Download Button Special Styling */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            border-radius: 12px;
            font-weight: 500;
            padding: 0.6rem 1.5rem;
            box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
        }
        
        .stDownloadButton > button:hover {
            box-shadow: 0 4px 16px rgba(16, 185, 129, 0.4);
            transform: translateY(-2px);
        }
        
        /* Input Field Styling */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > select,
        .stTextArea > div > div > textarea {
            border-radius: 10px;
            border: 2px solid #e2e8f0;
            padding: 0.6rem 0.8rem;
            transition: all 0.3s ease;
            font-size: 0.95rem;
        }
        
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        /* Selectbox Styling */
        .stSelectbox > div > div {
            border-radius: 10px;
        }
        
        /* Metric Card Styling */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
            color: #1e293b;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 0.85rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        [data-testid="stMetricDelta"] {
            font-size: 0.9rem;
        }
        
        /* DataFrame Styling */
        .dataframe {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            border: 1px solid #e2e8f0;
        }
        
        .dataframe thead tr th {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            color: #475569;
            font-weight: 600;
            padding: 12px;
            font-size: 0.9rem;
        }
        
        .dataframe tbody tr:hover {
            background-color: #f8fafc;
        }
        
        /* Alert/Info Box Styling */
        .stAlert {
            border-radius: 12px;
            padding: 1rem 1.25rem;
            border: none;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        }
        
        .stSuccess {
            background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
            border-left: 4px solid #10b981;
        }
        
        .stInfo {
            background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
            border-left: 4px solid #3b82f6;
        }
        
        .stWarning {
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            border-left: 4px solid #f59e0b;
        }
        
        .stError {
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
            border-left: 4px solid #ef4444;
        }
        
        /* Expander Styling */
        .streamlit-expanderHeader {
            border-radius: 10px;
            background-color: #f8fafc;
            font-weight: 500;
            color: #475569;
        }
        
        .streamlit-expanderHeader:hover {
            background-color: #f1f5f9;
        }
        
        /* File Uploader Styling */
        [data-testid="stFileUploader"] {
            border: 2px dashed #cbd5e1;
            border-radius: 12px;
            padding: 2rem;
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            transition: all 0.3s ease;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: #667eea;
            background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        }
        
        /* Divider Styling */
        hr {
            margin: 2rem 0;
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
        }
        
        /* Card/Container Styling */
        .element-container {
            transition: all 0.3s ease;
        }
        
        /* Checkbox Styling */
        .stCheckbox {
            padding: 0.5rem 0;
        }
        
        /* Radio Button Styling */
        .stRadio > label {
            font-weight: 500;
            color: #475569;
        }
        
        /* Spinner Styling */
        .stSpinner > div {
            border-color: #667eea transparent #667eea transparent;
        }
        
        /* Remove link underlines */
        a {
            text-decoration: none !important;
            color: #667eea;
            font-weight: 500;
        }
        
        a:hover {
            color: #764ba2;
        }
        
        /* Section Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #1e293b;
            font-weight: 600;
        }
        
        /* Improve overall spacing */
        .block-container > div {
            padding-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'excluded_wells' not in st.session_state:
        st.session_state.excluded_wells = {}
    
    # Process uploaded Excel file if present
    if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None and not st.session_state.data_loaded:
        try:
            uploaded_file = st.session_state.uploaded_file
            xls = pd.ExcelFile(uploaded_file)
            
            if 'Pre' not in xls.sheet_names or 'Post' not in xls.sheet_names:
                st.error("Excel file must contain 'Pre' and 'Post' sheets")
                st.session_state.uploaded_file = None
                st.stop()
            
            df_pre, avg_pre = process_sheet(xls, 'Pre')
            df_post, avg_post = process_sheet(xls, 'Post')
            
            if df_pre is None or df_post is None:
                st.error("Failed to process data sheets")
                st.session_state.uploaded_file = None
                st.stop()
            
            fold_change = (avg_post / avg_pre).round(2)
            
            labels_matrix = np.empty((8, 12), dtype=object)
            labels_matrix[:] = ""
            
            if 'Labels' in xls.sheet_names:
                try:
                    df_labels = pd.read_excel(xls, sheet_name='Labels', index_col=0)
                    for row_idx, row_letter in enumerate('ABCDEFGH'):
                        if row_letter in df_labels.index:
                            for col_idx in range(12):
                                col_name = str(col_idx + 1)
                                if col_name in df_labels.columns:
                                    label = df_labels.loc[row_letter, col_name]
                                    if pd.notna(label) and label != '':
                                        labels_matrix[row_idx, col_idx] = str(label)
                except Exception as e:
                    st.warning(f"Could not load labels: {e}")
            
            export_data = []
            fc_matrix = np.empty((8, 12))
            fc_matrix[:] = np.nan
            
            for row_idx, row_letter in enumerate("ABCDEFGH"):
                for col_idx in range(12):
                    well_name = f"{row_letter}{col_idx+1}"
                    if well_name in avg_pre.index:
                        label = labels_matrix[row_idx, col_idx]
                        pre_val = avg_pre[well_name]
                        post_val = avg_post[well_name]
                        fc_val = fold_change[well_name]
                        fc_matrix[row_idx, col_idx] = fc_val
                        export_data.append({
                            'Well': well_name,
                            'Label': label,
                            'Pre Avg': pre_val,
                            'Post Avg': post_val,
                            'Fold Change': fc_val
                        })
            
            export_df = pd.DataFrame(export_data)
            
            st.session_state.update({
                'export_df': export_df,
                'df_pre': df_pre,
                'df_post': df_post,
                'labels_matrix': labels_matrix,
                'fc_matrix': fc_matrix,
                'data_loaded': True
            })
            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.session_state.uploaded_file = None
            import traceback
            st.code(traceback.format_exc())
            st.stop()
    
    # Show welcome page if no data is loaded
    if not st.session_state.data_loaded:
        # WELCOME PAGE
        st.markdown("""
            <div style="text-align: center; padding: 2rem 1rem 1.5rem 1rem;">
                <div style="display: inline-block; padding: 0.75rem 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            border-radius: 12px; margin-bottom: 1rem; box-shadow: 0 4px 16px rgba(102, 126, 234, 0.25);">
                    <h1 style="font-size: 2rem; font-weight: 700; margin: 0; color: white; letter-spacing: -0.01em;">
                        GloSensor Analysis
                    </h1>
                </div>
                <p style="font-size: 1rem; color: #64748b; margin: 0.5rem 0 0 0; font-weight: 400;">
                    Professional cAMP biosensor data analysis
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Main content area
        col_spacer1, col_main, col_spacer2 = st.columns([1, 2.5, 1])
        
        with col_main:
            # Upload Excel Section
            st.markdown("""
                <div style="background: white; 
                            padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0;
                            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08); border: 1px solid #e2e8f0;">
                    <h2 style="color: #1e293b; margin: 0 0 0.5rem 0; font-size: 1.3rem; font-weight: 600;">
                        Upload Data
                    </h2>
                    <p style="color: #64748b; margin: 0 0 1rem 0; font-size: 0.9rem; line-height: 1.5;">
                        Import an Excel file containing Pre, Post, and Labels sheets
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose an Excel file (.xlsx or .xls)",
                type=['xlsx', 'xls'],
                help="Your Excel file must contain three sheets: 'Pre', 'Post', and 'Labels'",
                key="welcome_excel_upload"
            )
            
            if uploaded_file:
                st.session_state.uploaded_file = uploaded_file
                st.rerun()
            
            # Template download section
            st.markdown("""
                <div style="background: #fef9e7; 
                            padding: 1.25rem; border-radius: 10px; margin: 1.5rem 0;
                            border-left: 3px solid #f59e0b;">
                    <h3 style="color: #92400e; margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 600;">
                        Need a template?
                    </h3>
                    <p style="color: #78350f; margin: 0; font-size: 0.85rem; line-height: 1.5;">
                        Download our Excel template with the correct structure
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Create and provide template download
            template_buffer = BytesIO()
            
            # Create template data
            labels_data = {}
            for col_num in range(1, 13):
                labels_data[str(col_num)] = [''] * 8
            labels_template = pd.DataFrame(labels_data, index=list('ABCDEFGH'))
            
            time_points = [0, 5, 10, 15, 20, 25, 30]
            wells = []
            for row in "ABCDEFGH":
                for col in range(1, 13):
                    wells.append(f"{row}{col}")
            
            pre_template = pd.DataFrame({'Time (min)': time_points})
            for well in wells:
                pre_template[well] = 0.0
            
            post_template = pd.DataFrame({'Time (min)': time_points})
            for well in wells:
                post_template[well] = 0.0
            
            with pd.ExcelWriter(template_buffer, engine='openpyxl') as writer:
                labels_template.to_excel(writer, sheet_name='Labels', index=True)
                pre_template.to_excel(writer, sheet_name='Pre', index=False)
                post_template.to_excel(writer, sheet_name='Post', index=False)
            
            st.download_button(
                label="Download Excel Template",
                data=template_buffer.getvalue(),
                file_name="glosensor_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            # Divider
            st.markdown("""
                <div style="text-align: center; margin: 2rem 0 1.5rem 0;">
                    <div style="height: 1px; background: #e2e8f0;"></div>
                    <span style="background: #f8fafc; padding: 0.25rem 1rem; position: relative; top: -0.75rem; 
                                 color: #94a3b8; font-size: 0.8rem; font-weight: 500;">OR</span>
                </div>
            """, unsafe_allow_html=True)
            
            # Load saved analysis section
            st.markdown("""
                <div style="background: white; 
                            padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
                            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08); border: 1px solid #e2e8f0;">
                    <h2 style="color: #1e293b; margin: 0 0 0.5rem 0; font-size: 1.3rem; font-weight: 600;">
                        Continue Previous Work
                    </h2>
                    <p style="color: #64748b; margin: 0 0 1rem 0; font-size: 0.9rem; line-height: 1.5;">
                        Load a previously saved analysis session (.gsa file)
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            analysis_file = st.file_uploader(
                "Upload Saved Analysis (.gsa)",
                type=['gsa'],
                help="Load a previously saved analysis session",
                key="welcome_gsa_upload"
            )
            
            if analysis_file:
                with st.spinner("Loading analysis..."):
                    state = load_analysis_state(analysis_file)
                    if state:
                        st.session_state.update({
                            'export_df': state['export_df'],
                            'df_pre': state['df_pre'],
                            'df_post': state['df_post'],
                            'labels_matrix': state['labels_matrix'],
                            'fc_matrix': state['fc_matrix'],
                            'excluded_wells': state['excluded_wells'],
                            'data_loaded': True,
                            **state['meta']
                        })
                        st.success("Analysis loaded successfully")
                        st.rerun()
        
        # Don't show the rest of the app
        return
    
    # SIDEBAR - Only shown when data is loaded
    with st.sidebar:
        st.markdown("""
            <div style="background: #667eea; 
                        padding: 1rem; border-radius: 8px; text-align: center; margin-bottom: 1rem;">
                <h2 style="color: white; margin: 0; font-size: 1.2rem; font-weight: 600;">GloSensor Analysis</h2>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Experiment Info")
        experiment_name = st.text_input("Experiment Name*", 
                                       value=st.session_state.get('experiment_name', 'My Experiment'),
                                       help="Give your experiment a unique name")
        name_input = st.text_input("Investigator*", 
                                   value=st.session_state.get('name_input', 'Researcher'),
                                   help="Your name or lab member name")
        
        st.session_state.experiment_name = experiment_name
        st.session_state.name_input = name_input
        
        st.markdown("---")
        st.markdown("### Start Over")
        if st.button("Return to Welcome", use_container_width=True):
            st.session_state.data_loaded = False
            st.session_state.clear()
            st.rerun()
    
    if st.session_state.data_loaded:
        export_df = st.session_state.export_df
        df_pre = st.session_state.df_pre
        df_post = st.session_state.df_post
        labels_matrix = st.session_state.labels_matrix
        fc_matrix = st.session_state.fc_matrix
    
    # Show analysis tabs when data is loaded
    if not st.session_state.data_loaded:
        # This shouldn't happen as we return from welcome page, but just in case
        return
    
    tab1, tab2, tab3, tab4 = st.tabs(["Experimental Setup", "Well Labels", "Analysis Results", "Export"])
    
    with tab1:
        st.markdown("""
            <div style="text-align: center; padding: 2rem 0 1.5rem 0;">
                <h1 style="font-size: 2.2rem; font-weight: 700; 
                           background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                           -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                           margin: 0;">🧬 Experimental Setup</h1>
                <p style="font-size: 1rem; color: #64748b; margin: 0.75rem 0 0 0; font-weight: 400;">
                    Configure your assay parameters • All fields marked with * are required
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Step 1: Assay Method Selection (Always visible)
        st.markdown("""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #f5f3ff 100%); 
                        padding: 1rem 1.25rem; border-radius: 12px; margin: 1rem 0 1rem 0; 
                        border-left: 4px solid #667eea; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
                <h3 style="color: #1e293b; font-size: 1.1rem; font-weight: 600; margin: 0 0 0.25rem 0;">
                    🔬 Assay Method
                </h3>
                <p style="color: #64748b; font-size: 0.9rem; margin: 0; line-height: 1.5;">
                    Choose your experimental approach
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        col_method1, col_method2 = st.columns([1, 2])
        with col_method1:
            assay_method = st.radio(
                "Select Method*",
                ["Direct method", "Transfer method"],
                index=0 if st.session_state.get('assay_method', 'Direct method') == 'Direct method' else 1,
                key="assay_method_radio",
                help="Direct: cells seeded directly into assay plate. Transfer: cells grown in larger vessel then transferred"
            )
            st.session_state.assay_method = assay_method
        
        with col_method2:
            if assay_method == "Direct method":
                st.info("📌 **Direct Method:** Cells are seeded directly into 96-well assay plates and transfected in place")
                vessel_type = '96-well'
                st.session_state.vessel_type = vessel_type
            else:
                st.info("📌 **Transfer Method:** Cells are grown in a larger culture vessel, transfected, then transferred to 96-well assay plates")
        
        # Only show rest of form if assay method is selected
        if not assay_method:
            st.info("👆 Please select an assay method to continue")
            return
        
        # Step 2: Transfection Direction (Appears after assay method selected)
        st.markdown("""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #f5f3ff 100%); 
                        padding: 1rem 1.25rem; border-radius: 12px; margin: 1rem 0 1rem 0; 
                        border-left: 4px solid #667eea; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
                <h3 style="color: #1e293b; font-size: 1.1rem; font-weight: 600; margin: 0 0 0.25rem 0;">
                    🧫 Transfection Direction
                </h3>
                <p style="color: #64748b; font-size: 0.9rem; margin: 0; line-height: 1.5;">
                    Choose when transfection occurs relative to cell seeding
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        col_dir1, col_dir2 = st.columns([1, 2])
        with col_dir1:
            transfection_direction = st.radio(
                "Select Direction*",
                ["Forward transfection", "Reverse transfection"],
                index=0 if st.session_state.get('transfection_direction', 'Forward transfection') == 'Forward transfection' else 1,
                key="transfection_direction_radio",
                help="Forward: cells seeded first, then transfected. Reverse: seeded and transfected simultaneously"
            )
            st.session_state.transfection_direction = transfection_direction
        
        with col_dir2:
            if transfection_direction == "Forward transfection":
                st.info("📌 **Forward Transfection:** Seed cells → Wait for attachment → Transfect")
            else:
                st.success("📌 **Reverse Transfection:** Seeding and transfection occur simultaneously (0 hrs wait time)")
                st.session_state.hrs_to_transfection = "0 (simultaneous)"
        
        # Only show rest of form if transfection direction is selected
        if not transfection_direction:
            st.info("👆 Please select a transfection direction to continue")
            return
        
        # Now show the rest of the form based on selections
        st.markdown("---")
        
        # Cell Seeding
        st.markdown("""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #f5f3ff 100%); 
                        padding: 1rem 1.25rem; border-radius: 12px; margin: 1rem 0 1rem 0; 
                        border-left: 4px solid #667eea; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
                <h3 style="color: #1e293b; font-size: 1.1rem; font-weight: 600; margin: 0 0 0.25rem 0;">
                    Cell Seeding
                </h3>
                <p style="color: #64748b; font-size: 0.9rem; margin: 0; line-height: 1.5;">
                    Define your cell culture vessel, seeding density, and culture medium
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        col_seed1, col_seed2, col_seed3 = st.columns(3)
        
        with col_seed1:
            if assay_method == "Transfer method":
                vessel_options = list(VESSEL_PROTOCOLS.keys())
                current_vessel = st.session_state.get('vessel_type', '96-well')
                vessel_type = st.selectbox(
                    "Culture Vessel*",
                    vessel_options,
                    index=vessel_options.index(current_vessel) if current_vessel in vessel_options else 0,
                    help="Select your culture vessel for initial cell growth",
                    key="vessel_type_select"
                )
                st.session_state.vessel_type = vessel_type
            else:
                st.markdown("**Vessel Type**")
                st.markdown("🧪 96-well plate")
        
        default_cells = {
            '96-well': 50000, '24-well': 100000, '12-well': 200000,
            '6-well': 400000, '60-mm': 1000000, '10-cm': 2000000
        }
        
        with col_seed2:
            current_cells = st.session_state.get('cells_seeded', '')
            if current_cells and current_cells != '':
                try:
                    current_cells_val = int(current_cells.replace(',', ''))
                except:
                    current_cells_val = default_cells.get(vessel_type, 50000)
            else:
                current_cells_val = default_cells.get(vessel_type, 50000)
            
            if assay_method == "Transfer method":
                label_text = "Cells Seeded*"
                help_text = f"Total cells seeded in {vessel_type}"
            else:
                label_text = "Cells Seeded (per well)*"
                help_text = "Cells seeded per well of 96-well plate"
            
            cells_seeded = st.number_input(
                label_text,
                min_value=1000, max_value=10000000, value=current_cells_val, step=1000,
                help=help_text,
                key="cells_seeded_number", format="%d"
            )
            st.session_state.cells_seeded = f"{cells_seeded:,}"
        
        with col_seed3:
            vessel_info = VESSEL_PROTOCOLS[vessel_type]
            default_media = vessel_info['plating_medium']
            media_volume = st.text_input(
                "Culture Medium Volume*",
                value=st.session_state.get('media_volume', default_media),
                help=f"Media volume for cell culture. Typical: {default_media}",
                key="media_volume_input"
            )
            st.session_state.media_volume = media_volume
        
        # Cell Attachment Period (only for Forward)
        if transfection_direction == "Forward transfection":
            st.markdown("""
                <div style="background: linear-gradient(135deg, #f0f9ff 0%, #f5f3ff 100%); 
                            padding: 1rem 1.25rem; border-radius: 12px; margin: 1rem 0 1rem 0; 
                            border-left: 4px solid #667eea; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
                    <h3 style="color: #1e293b; font-size: 1.1rem; font-weight: 600; margin: 0 0 0.25rem 0;">
                        Cell Attachment Period
                    </h3>
                    <p style="color: #64748b; font-size: 0.9rem; margin: 0; line-height: 1.5;">
                        Time between seeding and transfection
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            current_hrs = st.session_state.get('hrs_to_transfection', '')
            if current_hrs and current_hrs != '':
                try:
                    current_hrs_val = int(current_hrs)
                except:
                    current_hrs_val = 24
            else:
                current_hrs_val = 24
            
            hrs_to_transfection = st.number_input(
                "Hours from Seeding to Transfection*",
                min_value=0, max_value=168, value=current_hrs_val, step=1,
                help="Allow cells to attach and reach appropriate confluence",
                key="hrs_to_transfection_number"
            )
            st.session_state.hrs_to_transfection = str(hrs_to_transfection)
        
        # Transfection Configuration
        st.markdown("""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #f5f3ff 100%); 
                        padding: 1rem 1.25rem; border-radius: 12px; margin: 1rem 0 1rem 0; 
                        border-left: 4px solid #667eea; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
                <h3 style="color: #1e293b; font-size: 1.1rem; font-weight: 600; margin: 0 0 0.25rem 0;">
                    Transfection Configuration
                </h3>
                <p style="color: #64748b; font-size: 0.9rem; margin: 0; line-height: 1.5;">
                    Configure transfection reagents and DNA amounts
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Transfection Agent
        transfection_options = ["Lipofectamine 2000", "Polyethylenimine (PEI)", "Other"]
        current_agent = st.session_state.get('transfection_agent', 'Lipofectamine 2000')
        if current_agent not in transfection_options:
            current_agent = "Other"
        
        col_agent1, col_agent2 = st.columns(2)
        with col_agent1:
            transfection_agent = st.selectbox(
                "Transfection Agent*",
                transfection_options,
                index=transfection_options.index(current_agent),
                key="transfection_agent_select"
            )
        
        with col_agent2:
            if transfection_agent == "Other":
                transfection_agent_custom = st.text_input(
                    "Specify agent:",
                    value=st.session_state.get('transfection_agent_custom', ''),
                    key="transfection_agent_custom_input"
                )
                st.session_state.transfection_agent = transfection_agent_custom if transfection_agent_custom else "Other"
            else:
                st.session_state.transfection_agent = transfection_agent
        
        # DNA and Reagent Amounts
        st.markdown('<p style="color: #1e293b; font-size: 0.9rem; font-weight: 600; margin: 1rem 0 0.5rem 0;">🧬 DNA & Transfection Reagent</p>', unsafe_allow_html=True)
        
        vessel_info = VESSEL_PROTOCOLS[vessel_type]
        dna_amount_from_vessel = vessel_info['dna_amount'].replace(' μg', '').replace('μg', '')
        try:
            base_dna_ug = float(dna_amount_from_vessel)
            default_biosensor = int(base_dna_ug * 1000 * 0.5)
        except:
            default_biosensor = 100
        
        col_dna1, col_dna2, col_dna3 = st.columns(3)
        
        with col_dna1:
            current_biosensor = st.session_state.get('biosensor_amount', '')
            if current_biosensor and current_biosensor != '':
                try:
                    current_biosensor_val = int(current_biosensor)
                except:
                    current_biosensor_val = default_biosensor
            else:
                current_biosensor_val = default_biosensor
            
            amount_label = "Biosensor DNA (ng)*"
            amount_help = f"Amount per well" if assay_method == "Direct method" else f"Total amount for {vessel_type}"
            
            biosensor_amount = st.number_input(
                amount_label,
                min_value=1, max_value=100000, value=current_biosensor_val, step=10,
                help=amount_help,
                key="biosensor_amount_number"
            )
            st.session_state.biosensor_amount = str(biosensor_amount)
        
        with col_dna2:
            current_gpcr = st.session_state.get('gpcr_amount', '')
            if current_gpcr and current_gpcr != '':
                try:
                    current_gpcr_val = int(current_gpcr)
                except:
                    current_gpcr_val = default_biosensor
            else:
                current_gpcr_val = default_biosensor
            
            gpcr_amount = st.number_input(
                "GPCR DNA (ng)*",
                min_value=1, max_value=100000, value=current_gpcr_val, step=10,
                help=amount_help,
                key="gpcr_amount_number"
            )
            st.session_state.gpcr_amount = str(gpcr_amount)
        
        with col_dna3:
            ratio_options = ["1:1", "1:2", "1:3", "1:4", "1:5", "1:6"]
            current_ratio = st.session_state.get('agent_amount_ratio', '1:3')
            if current_ratio not in ratio_options:
                current_ratio = '1:3'
            
            agent_amount_ratio = st.selectbox(
                "DNA to Transfection Reagent Ratio*",
                options=ratio_options,
                index=ratio_options.index(current_ratio),
                help="Ratio of DNA (μg) to Transfection Reagent (μl). For example, 1:3 means 1 μg DNA to 3 μl reagent.",
                key="agent_amount_ratio_input"
            )
            st.session_state.agent_amount_ratio = agent_amount_ratio
        
        # Transfection Mix Preparation
        st.markdown('<p style="color: #1e293b; font-size: 0.9rem; font-weight: 600; margin: 1rem 0 0.5rem 0;">⚗️ Transfection Mix Preparation</p>', unsafe_allow_html=True)
        
        col_prep1, col_prep2 = st.columns(2)
        
        with col_prep1:
            serum_free_medium = st.text_input(
                "Serum-Free Medium (μl)*",
                value=st.session_state.get('serum_free_medium', ''),
                help="Volume used for diluting DNA and transfection reagent",
                key="serum_free_medium_input",
                placeholder="e.g., 25"
            )
            st.session_state.serum_free_medium = serum_free_medium
        
        with col_prep2:
            current_incubation = st.session_state.get('dna_reagent_incubation', '')
            if current_incubation and current_incubation != '':
                try:
                    current_incubation_val = int(current_incubation)
                except:
                    current_incubation_val = 20
            else:
                current_incubation_val = 20
            
            dna_reagent_incubation = st.number_input(
                "DNA + Reagent Incubation (min)*",
                min_value=0, max_value=60, value=current_incubation_val, step=1,
                help="Time to incubate DNA-reagent complex before adding to cells",
                key="dna_reagent_incubation_number"
            )
            st.session_state.dna_reagent_incubation = str(dna_reagent_incubation)
        
        # Post-Transfection Timeline
        st.markdown("""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #f5f3ff 100%); 
                        padding: 1rem 1.25rem; border-radius: 12px; margin: 1rem 0 1rem 0; 
                        border-left: 4px solid #667eea; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
                <h3 style="color: #1e293b; font-size: 1.1rem; font-weight: 600; margin: 0 0 0.25rem 0;">
                    Post-Transfection Timeline
                </h3>
                <p style="color: #64748b; font-size: 0.9rem; margin: 0; line-height: 1.5;">
                    Define timing between transfection and assay
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        if assay_method == "Transfer method":
            col_time1, col_time2 = st.columns(2)
            
            with col_time1:
                current_transfer = st.session_state.get('hrs_to_transfer', '')
                if current_transfer and current_transfer != '':
                    try:
                        current_transfer_val = int(current_transfer)
                    except:
                        current_transfer_val = 24
                else:
                    current_transfer_val = 24
                
                hrs_to_transfer = st.number_input(
                    "Transfection → Transfer to 96-well (hrs)*",
                    min_value=0, max_value=168, value=current_transfer_val, step=1,
                    help="Hours after transfection before transferring cells to assay plate",
                    key="hrs_to_transfer_number"
                )
                st.session_state.hrs_to_transfer = str(hrs_to_transfer)
            
            with col_time2:
                current_incubation = st.session_state.get('incubation_time', '')
                if current_incubation and current_incubation != '':
                    try:
                        current_incubation_val = int(current_incubation)
                    except:
                        current_incubation_val = 24
                else:
                    current_incubation_val = 24
                
                incubation_time = st.number_input(
                    "Transfer → Assay (hrs)*",
                    min_value=0, max_value=168, value=current_incubation_val, step=1,
                    help="Hours after transfer before running assay",
                    key="incubation_time_number"
                )
                st.session_state.incubation_time = str(incubation_time)
            
            # Cells transferred - this is the LAST step before assay
            st.markdown('<p style="color: #1e293b; font-size: 0.9rem; font-weight: 600; margin: 1rem 0 0.5rem 0;">📋 Transfer to Assay Plate</p>', unsafe_allow_html=True)
            
            default_transfer = {
                '96-well': 20000, '24-well': 40000, '12-well': 80000,
                '6-well': 200000, '60-mm': 500000, '10-cm': 1000000
            }
            current_transferred = st.session_state.get('cells_transferred', '')
            if current_transferred and current_transferred != '':
                try:
                    current_transferred_val = int(current_transferred.replace(',', ''))
                except:
                    current_transferred_val = default_transfer.get(vessel_type, 20000)
            else:
                current_transferred_val = default_transfer.get(vessel_type, 20000)
            
            cells_transferred = st.number_input(
                "Cells Transferred per Well of 96-well Plate*",
                min_value=1000, max_value=10000000, value=current_transferred_val, step=1000,
                help=f"Number of cells transferred to each well. Typical: {default_transfer.get(vessel_type, 20000):,}",
                key="cells_transferred_number", format="%d"
            )
            st.session_state.cells_transferred = f"{cells_transferred:,}"
        else:
            # Direct method - just time to assay
            current_assay = st.session_state.get('hrs_to_assay', '')
            if current_assay and current_assay != '':
                try:
                    current_assay_val = int(current_assay)
                except:
                    current_assay_val = 48
            else:
                current_assay_val = 48
            
            hrs_to_assay = st.number_input(
                "Transfection → Assay (hrs)*",
                min_value=0, max_value=168, value=current_assay_val, step=1,
                help="Hours after transfection before running assay",
                key="hrs_to_assay_number"
            )
            st.session_state.hrs_to_assay = str(hrs_to_assay)
            # Clear transfer-specific fields
            st.session_state.hrs_to_transfer = ''
            st.session_state.cells_transferred = ''
            st.session_state.incubation_time = ''
        
        # Protocol Documentation
        st.markdown("""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #f5f3ff 100%); 
                        padding: 1rem 1.25rem; border-radius: 12px; margin: 1rem 0 1rem 0; 
                        border-left: 4px solid #667eea; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
                <h3 style="color: #1e293b; font-size: 1.1rem; font-weight: 600; margin: 0 0 0.25rem 0;">
                    Protocol Documentation
                </h3>
                <p style="color: #64748b; font-size: 0.9rem; margin: 0; line-height: 1.5;">
                    Add detailed methods and experimental notes (optional)
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        col_doc1, col_doc2 = st.columns(2)
        
        with col_doc1:
            methods_text = st.text_area(
                "📝 Detailed Methods",
                value=st.session_state.get('methods_text', ''),
                height=120,
                help="Describe your experimental protocol in detail",
                key="methods_text_input",
                placeholder="Enter step-by-step protocol details..."
            )
            st.session_state.methods_text = methods_text
        
        with col_doc2:
            notes_text = st.text_area(
                "📌 Additional Notes",
                value=st.session_state.get('notes_text', ''),
                height=120,
                help="Any observations, deviations, or important notes",
                key="notes_text_input",
                placeholder="Enter any additional observations or notes..."
            )
            st.session_state.notes_text = notes_text
        
    with tab2:
        st.header("🏷️ Well Labels Configuration")
        st.markdown("**Tip:** Wells with the same label will be treated as replicates in the analysis")
        st.markdown("---")
        
        cols = st.columns(13)
        cols[0].write("")
        for col_idx in range(12):
            with cols[col_idx + 1]:
                st.markdown(f"**{col_idx+1}**")
        
        for row_idx, row_letter in enumerate("ABCDEFGH"):
            cols = st.columns(13)
            with cols[0]:
                st.markdown(f"**{row_letter}**")
            
            for col_idx in range(12):
                well_name = f"{row_letter}{col_idx+1}"
                default_label = labels_matrix[row_idx, col_idx]
                with cols[col_idx + 1]:
                    label = st.text_input(
                        f"{well_name}",
                        value=default_label,
                        key=f"tab2_label_{well_name}",
                        label_visibility="collapsed"
                    )
                    labels_matrix[row_idx, col_idx] = label if label else ""
        
        st.session_state.labels_matrix = labels_matrix
    
    export_data = []
    for row_idx, row_letter in enumerate("ABCDEFGH"):
        for col_idx in range(12):
            well_name = f"{row_letter}{col_idx+1}"
            well_data = export_df[export_df['Well'] == well_name]
            if not well_data.empty:
                label = labels_matrix[row_idx, col_idx]
                export_data.append({
                    'Well': well_name,
                    'Label': label,
                    'Pre Avg': well_data['Pre Avg'].values[0],
                    'Post Avg': well_data['Post Avg'].values[0],
                    'Fold Change': well_data['Fold Change'].values[0]
                })
    
    export_df = pd.DataFrame(export_data)
    st.session_state.export_df = export_df
    
    replicate_df = analyze_replicates(export_df, st.session_state.excluded_wells)
    
    with tab3:
        st.markdown("""
            <div style="text-align: center; padding: 2rem 0 1.5rem 0;">
                <h1 style="font-size: 2.2rem; font-weight: 700; 
                           background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                           -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                           margin: 0;">📊 Analysis Results</h1>
                <p style="font-size: 1rem; color: #64748b; margin: 0.75rem 0 0 0; font-weight: 400;">
                    Review your experimental data and statistical analysis
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Calculate exclusion stats
        total_excluded = sum(len(wells) for wells in st.session_state.excluded_wells.values())
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_wells = len(export_df)
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 0.75rem 1rem; border-radius: 10px; 
                            box-shadow: 0 3px 10px rgba(102, 126, 234, 0.3);">
                    <div style="color: rgba(255,255,255,0.85); font-size: 0.65rem; font-weight: 600; 
                                text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 0.3rem;">
                        Total Wells
                    </div>
                    <div style="color: white; font-size: 1.75rem; font-weight: 700; line-height: 1;">
                        {total_wells}
                    </div>
                    <div style="color: rgba(255,255,255,0.75); font-size: 0.7rem; margin-top: 0.25rem;">
                        Analyzed
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            active_wells = len(export_df[export_df['Fold Change'] > 2.0])
            percentage = f"{active_wells/len(export_df)*100:.1f}%" if len(export_df) > 0 else "0%"
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 0.75rem 1rem; border-radius: 10px; 
                            box-shadow: 0 3px 10px rgba(245, 87, 108, 0.3);">
                    <div style="color: rgba(255,255,255,0.85); font-size: 0.65rem; font-weight: 600; 
                                text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 0.3rem;">
                        Active Wells
                    </div>
                    <div style="color: white; font-size: 1.75rem; font-weight: 700; line-height: 1;">
                        {active_wells}
                    </div>
                    <div style="color: rgba(255,255,255,0.75); font-size: 0.7rem; margin-top: 0.25rem;">
                        {percentage} with FC > 2.0
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            replicate_count = len(replicate_df) if replicate_df is not None else 0
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                            padding: 0.75rem 1rem; border-radius: 10px; 
                            box-shadow: 0 3px 10px rgba(79, 172, 254, 0.3);">
                    <div style="color: rgba(255,255,255,0.85); font-size: 0.65rem; font-weight: 600; 
                                text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 0.3rem;">
                        Replicate Groups
                    </div>
                    <div style="color: white; font-size: 1.75rem; font-weight: 700; line-height: 1;">
                        {replicate_count}
                    </div>
                    <div style="color: rgba(255,255,255,0.75); font-size: 0.7rem; margin-top: 0.25rem;">
                        Detected
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 0.75rem 1rem; border-radius: 10px; 
                            box-shadow: 0 3px 10px rgba(250, 112, 154, 0.3);">
                    <div style="color: rgba(255,255,255,0.85); font-size: 0.65rem; font-weight: 600; 
                                text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 0.3rem;">
                        Excluded Wells
                    </div>
                    <div style="color: white; font-size: 1.75rem; font-weight: 700; line-height: 1;">
                        {total_excluded}
                    </div>
                    <div style="color: rgba(255,255,255,0.75); font-size: 0.7rem; margin-top: 0.25rem;">
                        Manually removed
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if replicate_df is not None:
            grouped = export_df[export_df['Label'].str.strip() != ''].groupby('Label')
            replicate_groups = {label: group for label, group in grouped if len(group) > 1}
            
            if replicate_groups:
                st.markdown('<h4 style="color: #1e293b; font-size: 1.05rem; font-weight: 600; margin-bottom: 0.5rem;">🧬 Replicate Groups</h4><p style="color: #64748b; font-size: 0.85rem; margin-bottom: 1rem;">Click on any group card to expand and manage wells</p>', unsafe_allow_html=True)
                
                # Initialize expanded_groups as empty set to start all collapsed
                if 'expanded_groups' not in st.session_state:
                    st.session_state.expanded_groups = set()
                
                for label in sorted(replicate_groups.keys()):
                    group = replicate_groups[label]
                    
                    if label not in st.session_state.excluded_wells:
                        st.session_state.excluded_wells[label] = []
                    
                    excluded_for_label = st.session_state.excluded_wells[label]
                    included_wells = group[~group['Well'].isin(excluded_for_label)]
                    
                    if len(included_wells) > 1:
                        mean_fc = included_wells['Fold Change'].mean()
                        std_fc = included_wells['Fold Change'].std()
                        cv_percent = (std_fc / mean_fc * 100) if mean_fc != 0 else 0
                    elif len(included_wells) == 1:
                        mean_fc = included_wells['Fold Change'].values[0]
                        std_fc = 0
                        cv_percent = 0
                    else:
                        mean_fc = 0
                        std_fc = 0
                        cv_percent = 0
                    
                    if cv_percent > 20:
                        cv_color = "#F8B4B4"
                        cv_border = "#FECACA"
                        cv_icon = "●"
                        cv_status = "High Variability"
                        status_bg = "#FEF2F2"
                    elif cv_percent > 10:
                        cv_color = "#FCD9A6"
                        cv_border = "#FDE8C7"
                        cv_icon = "●"
                        cv_status = "Moderate"
                        status_bg = "#FFFBEB"
                    else:
                        cv_color = "#A7F3D0"
                        cv_border = "#D1FAE5"
                        cv_icon = "●"
                        cv_status = "Excellent"
                        status_bg = "#F0FDF4"
                    
                    n_included = len(included_wells)
                    n_total = len(group)
                    mean_fc_display = f"{mean_fc:.3f}" if n_included > 0 else "—"
                    cv_display = f"{cv_percent:.1f}%"
                    
                    is_expanded = label in st.session_state.expanded_groups
                    expand_icon = "▼" if is_expanded else "▶"
                    
                    # Display the beautiful card HTML
                    st.markdown(f'''
                        <div style="background: linear-gradient(135deg, {status_bg} 0%, white 100%); 
                                    padding: 1rem 1.25rem; border-radius: 10px; margin-bottom: {"0.25rem" if is_expanded else "0.5rem"}; 
                                    border: 2px solid {cv_border}; cursor: pointer;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.05); transition: all 0.2s ease;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="flex: 1;">
                                    <div style="font-size: 1.1rem; font-weight: 600; color: #1e293b; margin-bottom: 0.3rem;">
                                        {expand_icon} {label}
                                    </div>
                                    <div style="display: flex; gap: 1.5rem; align-items: center; flex-wrap: wrap;">
                                        <span style="font-size: 0.85rem; color: #64748b;">
                                            <span style="color: {cv_color}; font-size: 1rem;">{cv_icon}</span> 
                                            CV: <strong style="color: #1e293b;">{cv_display}</strong>
                                        </span>
                                        <span style="font-size: 0.85rem; color: #64748b;">
                                            Wells: <strong style="color: #1e293b;">{n_included}/{n_total}</strong>
                                        </span>
                                        <span style="font-size: 0.85rem; color: #64748b;">
                                            Mean FC: <strong style="color: #1e293b;">{mean_fc_display}</strong>
                                        </span>
                                    </div>
                                </div>
                                <div style="font-size: 0.75rem; padding: 0.3rem 0.6rem; border-radius: 6px; 
                                            background-color: {cv_color}; color: #2D3748; font-weight: 600; margin-left: 1rem;">
                                    {cv_status}
                                </div>
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)
                    
                    # Create overlay button - positioned over the card
                    unique_key = f"btn_{label.replace(' ', '_').replace('.', '_')}"
                    if st.button(".", key=unique_key, use_container_width=True, help=f"Click to expand/collapse {label}"):
                        if label in st.session_state.expanded_groups:
                            st.session_state.expanded_groups.remove(label)
                        else:
                            st.session_state.expanded_groups.add(label)
                        st.rerun()
                    
                    # Style the button to be transparent overlay
                    st.markdown(f'''
                        <style>
                        button[kind="secondary"]:has(p:only-child) {{
                            opacity: 0.01 !important;
                            position: relative !important;
                            margin-top: -5.2rem !important;
                            height: 5rem !important;
                            cursor: pointer !important;
                            background: transparent !important;
                            border: none !important;
                            padding: 0 !important;
                        }}
                        button[kind="secondary"]:has(p:only-child):hover {{
                            opacity: 0.05 !important;
                            background: rgba(180, 199, 231, 0.1) !important;
                        }}
                        </style>
                    ''', unsafe_allow_html=True)
                    
                    # Show details if expanded
                    if is_expanded:
                        st.markdown(f'<div style="background-color: white; padding: 1rem; border-radius: 8px; margin-bottom: 0.75rem; border-left: 4px solid {cv_border}; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">', unsafe_allow_html=True)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        std_fc_display = f"{std_fc:.3f}" if n_included > 1 else "—"
                        
                        with col1:
                            st.markdown(f'<div style="background-color: #F8FBFE; padding: 0.5rem; border-radius: 8px; text-align: center;"><div style="color: #64748b; font-size: 0.7rem; font-weight: 500; text-transform: uppercase; margin-bottom: 0.2rem;">Mean FC</div><div style="color: #1e293b; font-size: 1.3rem; font-weight: 600;">{mean_fc_display}</div></div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown(f'<div style="background-color: #F8FBFE; padding: 0.5rem; border-radius: 8px; text-align: center;"><div style="color: #64748b; font-size: 0.7rem; font-weight: 500; text-transform: uppercase; margin-bottom: 0.2rem;">Std Dev</div><div style="color: #1e293b; font-size: 1.3rem; font-weight: 600;">{std_fc_display}</div></div>', unsafe_allow_html=True)
                        with col3:
                            st.markdown(f'<div style="background-color: #F8FBFE; padding: 0.5rem; border-radius: 8px; text-align: center;"><div style="color: #64748b; font-size: 0.7rem; font-weight: 500; text-transform: uppercase; margin-bottom: 0.2rem;">CV%</div><div style="color: #1e293b; font-size: 1.3rem; font-weight: 600;">{cv_display}</div></div>', unsafe_allow_html=True)
                        with col4:
                            st.markdown(f'<div style="background-color: #F8FBFE; padding: 0.5rem; border-radius: 8px; text-align: center;"><div style="color: #64748b; font-size: 0.7rem; font-weight: 500; text-transform: uppercase; margin-bottom: 0.2rem;">Wells (n)</div><div style="color: #1e293b; font-size: 1.3rem; font-weight: 600;">{n_included} / {n_total}</div></div>', unsafe_allow_html=True)
                        
                        st.markdown('<div style="background-color: #F8FBFE; padding: 0.5rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #667eea; box-shadow: 0 2px 8px rgba(0,0,0,0.04);"><span style="font-size: 0.85rem; color: #1e293b; font-weight: 500;">💡 Select Wells to Include in Analysis</span></div>', unsafe_allow_html=True)
                        
                        well_cols = st.columns(min(len(group), 4))
                        
                        for idx, (_, row) in enumerate(group.iterrows()):
                            well = row['Well']
                            fc = row['Fold Change']
                            is_included = well not in excluded_for_label
                            
                            with well_cols[idx % len(well_cols)]:
                                if is_included:
                                    card_bg = "#F0FDF4"
                                    border_color = "#A7F3D0"
                                    status_symbol = "◆"
                                else:
                                    card_bg = "#F9FAFB"
                                    border_color = "#E5E7EB"
                                    status_symbol = "◇"
                                
                                st.markdown(f'<div style="background-color: {card_bg}; padding: 0.5rem; border-radius: 6px; border: 2px solid {border_color}; margin-bottom: 0.4rem;"><div style="font-weight: 600; color: #1e293b; margin-bottom: 0.2rem; font-size: 0.85rem;">{status_symbol} {well}</div><div style="font-size: 0.8rem; color: #64748b;">FC: <span style="font-weight: 600;">{fc:.3f}</span></div></div>', unsafe_allow_html=True)
                                
                                include = st.checkbox(
                                    "Include" if is_included else "Excluded",
                                    value=is_included,
                                    key=f"include_{label}_{well}_v3",
                                    label_visibility="visible"
                                )
                                
                                if not include and well not in st.session_state.excluded_wells[label]:
                                    st.session_state.excluded_wells[label].append(well)
                                    st.rerun()
                                elif include and well in st.session_state.excluded_wells[label]:
                                    st.session_state.excluded_wells[label].remove(well)
                                    st.rerun()
                        
                        st.markdown('<div style="border-top: 1px solid #E8F4FD; padding-top: 0.5rem; margin-top: 0.5rem;"><span style="font-size: 0.8rem; color: #64748b; font-weight: 500;">QUICK ACTIONS</span></div>', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"◆ Include All Wells", key=f"include_all_{label}", use_container_width=True):
                                st.session_state.excluded_wells[label] = []
                                st.rerun()
                        with col2:
                            if st.button(f"◇ Exclude All Wells", key=f"exclude_all_{label}", use_container_width=True):
                                st.session_state.excluded_wells[label] = [row['Well'] for _, row in group.iterrows()]
                                st.rerun()
                        
                        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.header("💾 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("💾 Save Analysis")
            st.markdown("Save your work to continue later")
            
            meta = {
                'experiment_name': st.session_state.get('experiment_name', experiment_name),
                'name_input': st.session_state.get('name_input', name_input),
                'assay_method': st.session_state.get('assay_method', ''),
                'transfection_direction': st.session_state.get('transfection_direction', ''),
                'transfection_agent': st.session_state.get('transfection_agent', ''),
                'agent_amount_ratio': st.session_state.get('agent_amount_ratio', ''),
                'vessel_type': st.session_state.get('vessel_type', ''),
                'methods_text': st.session_state.get('methods_text', ''),
                'cells_seeded': st.session_state.get('cells_seeded', ''),
                'media_volume': st.session_state.get('media_volume', ''),
                'hrs_to_transfection': st.session_state.get('hrs_to_transfection', ''),
                'biosensor_amount': st.session_state.get('biosensor_amount', ''),
                'gpcr_amount': st.session_state.get('gpcr_amount', ''),
                'hrs_to_assay': st.session_state.get('hrs_to_assay', ''),
                'hrs_to_transfer': st.session_state.get('hrs_to_transfer', ''),
                'cells_transferred': st.session_state.get('cells_transferred', ''),
                'incubation_time': st.session_state.get('incubation_time', ''),
                'notes_text': st.session_state.get('notes_text', ''),
                'serum_free_medium': st.session_state.get('serum_free_medium', ''),
                'dna_reagent_incubation': st.session_state.get('dna_reagent_incubation', ''),
            }
            
            analysis_bytes = save_analysis_state(
                export_df, 
                df_pre, 
                df_post, 
                labels_matrix, 
                fc_matrix, 
                meta,
                st.session_state.excluded_wells
            )
            
            st.download_button(
                label="💾 Save Analysis File",
                data=analysis_bytes,
                file_name=f"{experiment_name}_analysis.gsa",
                mime="application/octet-stream",
                use_container_width=True,
                help="Save your current analysis to resume later"
            )
        
        with col2:
            st.subheader("📊 Excel Report")
            st.markdown("Download all data in Excel format")
            
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                export_df.to_excel(writer, sheet_name='Summary', index=False)
                if replicate_df is not None:
                    replicate_df.to_excel(writer, sheet_name='Replicates', index=False)
                df_pre.to_excel(writer, sheet_name='Pre', index=False)
                df_post.to_excel(writer, sheet_name='Post', index=False)
            
            st.download_button(
                label="📥 Download Excel Report",
                data=excel_buffer.getvalue(),
                file_name=f"{experiment_name}_glosensor_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col3:
            st.subheader("📄 Complete Export")
            st.markdown("PDF report + GSA file")
            
            # Check required fields based on method
            assay_method = st.session_state.get('assay_method', 'Direct method')
            transfection_direction = st.session_state.get('transfection_direction', '')
            
            base_fields = [
                st.session_state.get('assay_method'),
                st.session_state.get('transfection_direction'),
                st.session_state.get('transfection_agent'),
                st.session_state.get('agent_amount_ratio'),
                st.session_state.get('cells_seeded'),
                st.session_state.get('hrs_to_transfection'),
                st.session_state.get('biosensor_amount'),
                st.session_state.get('gpcr_amount'),
                st.session_state.get('serum_free_medium'),
                st.session_state.get('dna_reagent_incubation'),
            ]
            
            # Add media_volume check for Forward transfection
            if transfection_direction == "Forward transfection":
                base_fields.append(st.session_state.get('media_volume'))
            
            if assay_method == "Transfer method":
                required_fields_filled = all(base_fields + [
                    st.session_state.get('vessel_type'),
                    st.session_state.get('hrs_to_transfer'),
                    st.session_state.get('cells_transferred'),
                    st.session_state.get('incubation_time')
                ])
            else:
                required_fields_filled = all(base_fields + [
                    st.session_state.get('hrs_to_assay')
                ])
            
            if not required_fields_filled:
                st.warning("⚠️ Please fill in all required parameters in the Setup tab")
                st.button("📥 Generate Export", disabled=True, use_container_width=True)
            else:
                st.info("💡 All selections captured in both files")
                
                if st.button("📥 Generate Export", use_container_width=True, type="primary"):
                    with st.spinner("Generating export package..."):
                        try:
                            kinetics_path = create_individual_well_kinetics(df_pre, df_post, labels_matrix, fc_matrix)
                            heatmap_path = create_fold_change_heatmap(fc_matrix, labels_matrix)
                            
                            fig_files_info = [
                                ("Fold Change Heatmap", heatmap_path),
                                ("Individual Well Kinetics (96-Well Format)", kinetics_path)
                            ]
                            
                            meta = {
                                'experiment_name': st.session_state.get('experiment_name', experiment_name),
                                'name_input': st.session_state.get('name_input', name_input),
                                'assay_method': st.session_state.get('assay_method', ''),
                                'transfection_direction': st.session_state.get('transfection_direction', ''),
                                'transfection_agent': st.session_state.get('transfection_agent', ''),
                                'agent_amount_ratio': st.session_state.get('agent_amount_ratio', ''),
                                'vessel_type': st.session_state.get('vessel_type', ''),
                                'methods_text': st.session_state.get('methods_text', ''),
                                'cells_seeded': st.session_state.get('cells_seeded', ''),
                                'media_volume': st.session_state.get('media_volume', ''),
                                'hrs_to_transfection': st.session_state.get('hrs_to_transfection', ''),
                                'biosensor_amount': st.session_state.get('biosensor_amount', ''),
                                'gpcr_amount': st.session_state.get('gpcr_amount', ''),
                                'hrs_to_assay': st.session_state.get('hrs_to_assay', ''),
                                'hrs_to_transfer': st.session_state.get('hrs_to_transfer', ''),
                                'cells_transferred': st.session_state.get('cells_transferred', ''),
                                'incubation_time': st.session_state.get('incubation_time', ''),
                                'notes_text': st.session_state.get('notes_text', ''),
                                'serum_free_medium': st.session_state.get('serum_free_medium', ''),
                                'dna_reagent_incubation': st.session_state.get('dna_reagent_incubation', ''),
                                'replicate_df': replicate_df,
                                'excluded_wells': st.session_state.excluded_wells
                            }
                            
                            pdf_bytes = generate_pdf_bytes(
                                fig_files_info, labels_matrix, fc_matrix,
                                export_df, df_pre, df_post, meta
                            )
                            
                            analysis_bytes = save_analysis_state(
                                export_df, 
                                df_pre, 
                                df_post, 
                                labels_matrix, 
                                fc_matrix, 
                                meta,
                                st.session_state.excluded_wells
                            )
                            
                            try:
                                os.unlink(kinetics_path)
                                os.unlink(heatmap_path)
                            except:
                                pass
                            
                            if pdf_bytes:
                                st.success("✨ Export generated successfully!")
                                
                                st.markdown("""
                                    <div style="background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%); 
                                                padding: 1rem; border-radius: 10px; margin: 1rem 0; 
                                                border-left: 4px solid #4CAF50; text-align: center;">
                                        <h3 style="color: #2E7D32; margin: 0 0 0.5rem 0; font-size: 1.1rem;">
                                            ✅ Your files are ready to download!
                                        </h3>
                                        <p style="color: #388E3C; margin: 0; font-size: 0.9rem;">
                                            Click the buttons below to save your files
                                        </p>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                col_pdf, col_gsa = st.columns(2)
                                with col_pdf:
                                    st.markdown("**📄 Complete PDF Report**")
                                    st.download_button(
                                        label="⬇️ Download PDF Report",
                                        data=pdf_bytes,
                                        file_name=f"{experiment_name}_report.pdf",
                                        mime="application/pdf",
                                        use_container_width=True,
                                        type="primary"
                                    )
                                    st.caption("Complete analysis with all figures and data")
                                
                                with col_gsa:
                                    st.markdown("**💾 Analysis Save File**")
                                    st.download_button(
                                        label="⬇️ Download GSA File",
                                        data=analysis_bytes,
                                        file_name=f"{experiment_name}_analysis.gsa",
                                        mime="application/octet-stream",
                                        use_container_width=True,
                                        help="Save to resume later"
                                    )
                                    st.caption("Load this file to continue your work later")
                            else:
                                st.error("Failed to generate export")
                        except Exception as e:
                            st.error(f"Error generating files: {e}")
                            import traceback
                            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
