import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import tempfile
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, PageBreak, Image, Spacer
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER

def analyze_replicates(export_df):
    """Analyze replicates - wells with the same label"""
    # Group by label
    replicate_data = []
    
    grouped = export_df.groupby('Label')
    
    for label, group in grouped:
        if label and label.strip() != '':  # Only process non-empty labels
            n = len(group)
            if n > 1:  # Only include if there are actual replicates
                mean_fc = group['Fold Change'].mean()
                std_fc = group['Fold Change'].std()
                cv_percent = (std_fc / mean_fc * 100) if mean_fc != 0 else 0
                
                replicate_data.append({
                    'Label': label,
                    'n': n,
                    'Wells': ', '.join(group['Well'].tolist()),
                    'Mean FC': round(mean_fc, 3),
                    'Std Dev': round(std_fc, 3),
                    'CV%': round(cv_percent, 1),
                    'Individual FCs': ', '.join([f"{fc:.2f}" for fc in group['Fold Change']])
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
    
    # Set white background for the entire figure
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
        
        # Set white background for each subplot
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
                # Set white borders for non-highlighted wells
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
    
    # Set PDF metadata
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
    
    # Methods Section
    elements.append(Paragraph("<b>METHODS</b>", section_style))
    elements.append(Spacer(1, 15))
    
    methods_paragraph_style = ParagraphStyle(
        name='Methods',
        fontSize=11,
        leading=16,
        spaceAfter=12,
        alignment=0,  # Left alignment
        leftIndent=30,
        rightIndent=30
    )
    
    # Transfection details
    transfection_section = f"""
    <b>Experimental Timeline and Transfection Details:</b><br/>
    Number of cells seeded per well: {meta.get('cells_seeded', 'N/A')}<br/>
    Hours after seeding to transfection: {meta.get('hrs_to_transfection', 'N/A')} hrs<br/>
    Biosensor transfected per well: {meta.get('biosensor_amount', 'N/A')} ng<br/>
    GPCR transfected per well: {meta.get('gpcr_amount', 'N/A')} ng<br/>
    Hours after transfection to assay: {meta.get('hrs_to_assay', 'N/A')} hrs<br/>
    <br/>
    """
    
    elements.append(Paragraph(transfection_section, methods_paragraph_style))
    elements.append(Spacer(1, 10))
    
    # Get methods text from meta, or show placeholder if empty
    methods_content = meta.get('methods_text', '')
    if methods_content and methods_content.strip() != '':
        elements.append(Paragraph("<b>Protocol:</b><br/>" + methods_content.replace('\n', '<br/>'), methods_paragraph_style))
        elements.append(Spacer(1, 10))
    
    # Get notes text from meta if provided
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
        
        # Add replicate analysis after the Fold Change Heatmap
        if title == "Fold Change Heatmap":
            replicate_df = meta.get('replicate_df')
            if replicate_df is not None and not replicate_df.empty:
                elements.append(Paragraph("<b>REPLICATE ANALYSIS</b>", section_style))
                elements.append(Spacer(1, 10))
                
                # Create replicate table
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
                
                # Style the table with color coding based on CV%
                table_style = [
                    ("GRID", (0,0), (-1,-1), 0.5, colors.black),
                    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#34495E")),
                    ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
                    ("FONTSIZE", (0,0), (-1,-1), 8),
                    ("ALIGN", (0,0), (-1,-1), "CENTER"),
                    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ]
                
                # Add row colors based on CV%
                for idx, (_, row) in enumerate(replicate_df.iterrows(), start=1):
                    cv = row['CV%']
                    if cv > 20:
                        color = colors.HexColor("#ffcccc")  # Red
                    elif cv > 10:
                        color = colors.HexColor("#fff4cc")  # Yellow
                    else:
                        color = colors.HexColor("#ccffcc")  # Green
                    table_style.append(("BACKGROUND", (0,idx), (-1,idx), color))
                
                rep_table.setStyle(TableStyle(table_style))
                
                elements.append(rep_table)
                elements.append(Spacer(1, 10))
                
                legend_text = "Color coding: Green (CV < 10%, Good) | Yellow (CV 10-20%, Moderate) | Red (CV > 20%, High variability)"
                elements.append(Paragraph(legend_text, small_style))
                elements.append(Spacer(1, 20))
                elements.append(PageBreak())

    elements.append(Paragraph("<b>Raw Data Summary (Pre vs Post by Well)</b>", section_style))
    elements.append(Spacer(1, 10))
    
    wells = [col for col in df_pre.columns if col != 'Time (min)']
    time_points = [f"{t:.1f}" for t in df_pre['Time (min)']]
    
    # Calculate consistent column widths
    num_time_cols = len(time_points)
    well_col_width = 50  # Increased width for well name + label column
    time_col_width = 40  # Width for each time point column
    table_width = well_col_width + (num_time_cols * time_col_width)
    
    # Create column widths list
    col_widths = [well_col_width] + [time_col_width] * num_time_cols
    
    # Split wells into groups of 24 to fit on page
    wells_per_page = 24
    
    for well_idx in range(0, len(wells), wells_per_page):
        well_subset = wells[well_idx:well_idx+wells_per_page]
        
        # Create Pre table for this subset
        pre_header = [['PRE'], ['Well'] + time_points]
        pre_data = pre_header
        for well in well_subset:
            # Get label for this well
            row_letter = well[0]
            col_num = int(well[1:])
            row_idx = ord(row_letter) - ord('A')
            col_idx = col_num - 1
            label = labels_matrix[row_idx, col_idx]
            
            # Format well name with label
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
        
        # Create Post table for this subset
        post_header = [['POST'], ['Well'] + time_points]
        post_data = post_header
        for well in well_subset:
            # Get label for this well
            row_letter = well[0]
            col_num = int(well[1:])
            row_idx = ord(row_letter) - ord('A')
            col_idx = col_num - 1
            label = labels_matrix[row_idx, col_idx]
            
            # Format well name with label
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
        
        # Put tables side by side with fixed widths
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
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            padding: 1rem 0;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            padding-bottom: 2rem;
        }
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .success-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 1rem 0;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 1rem 2rem;
            font-size: 1.1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">🧪 GloSensor Assay Analysis Tool</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Professional analysis and reporting for cAMP bioluminescence assays</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.image("https://via.placeholder.com/300x80/1f77b4/ffffff?text=GloSensor+Analysis", use_container_width=True)
        st.markdown("---")
        
        st.header("📋 Experiment Information")
        experiment_name = st.text_input("Experiment Name*", "My Experiment", help="Give your experiment a unique name")
        name_input = st.text_input("Investigator*", "Researcher", help="Your name or lab member name")
        
        st.markdown("---")
        st.header("📁 Upload Data")
        uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx', 'xls'], help="File must contain 'Pre', 'Post', and 'Labels' sheets")
        
        if uploaded_file:
            st.success("✓ File uploaded successfully!")
        
        st.markdown("---")
        st.header("📄 Template")
        
        # Template generation
        template_buffer = BytesIO()
        with pd.ExcelWriter(template_buffer, engine='openpyxl') as writer:
            labels_data = {}
            for col_num in range(1, 13):
                labels_data[str(col_num)] = [''] * 8
            labels_template = pd.DataFrame(labels_data, index=list('ABCDEFGH'))
            labels_template.to_excel(writer, sheet_name='Labels', index=True)
            
            time_points = [0, 5, 10, 15, 20, 25, 30]
            wells = []
            for row in "ABCDEFGH":
                for col in range(1, 13):
                    wells.append(f"{row}{col}")
            
            pre_template = pd.DataFrame({'Time (min)': time_points})
            for well in wells:
                pre_template[well] = 0.0
            pre_template.to_excel(writer, sheet_name='Pre', index=False)
            
            post_template = pd.DataFrame({'Time (min)': time_points})
            for well in wells:
                post_template[well] = 0.0
            post_template.to_excel(writer, sheet_name='Post', index=False)
        
        st.download_button(
            label="📥 Download Template",
            data=template_buffer.getvalue(),
            file_name="glosensor_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    if uploaded_file is None:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("👈 **Get started:** Upload your Excel file using the sidebar", icon="ℹ️")
            
            with st.expander("📖 How to use this tool", expanded=True):
                st.markdown("""
                ### Quick Start Guide
                
                1. **Download the template** from the sidebar
                2. **Fill in your data:**
                   - `Labels` sheet: Sample names for each well
                   - `Pre` sheet: Baseline readings before treatment
                   - `Post` sheet: Readings after compound addition
                3. **Upload** your completed Excel file
                4. **Configure** experimental parameters
                5. **Generate** your analysis report
                
                ### File Format Requirements
                - Three sheets: **Labels**, **Pre**, and **Post**
                - Labels: 96-well plate format (rows A-H, columns 1-12)
                - Pre/Post: First column **Time (min)**, remaining columns: Well IDs
                - Minimum 3 timepoints required
                """)
            
            with st.expander("🔬 Features"):
                st.markdown("""
                - ✅ Automated fold change calculation
                - ✅ Replicate analysis with statistics
                - ✅ Interactive visualizations
                - ✅ Color-coded heatmaps
                - ✅ Individual well kinetics
                - ✅ Professional PDF reports
                - ✅ Excel export
                """)
        return
    
    # File uploaded - process data
    try:
        xls = pd.ExcelFile(uploaded_file)
        
        if 'Pre' not in xls.sheet_names or 'Post' not in xls.sheet_names:
            st.error("❌ Excel file must contain 'Pre' and 'Post' sheets")
            return
        
        df_pre, avg_pre = process_sheet(xls, 'Pre')
        df_post, avg_post = process_sheet(xls, 'Post')
        
        if df_pre is None or df_post is None:
            st.error("❌ Failed to process data sheets")
            return
        
        fold_change = (avg_post / avg_pre).round(2)
        
        # Load labels
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
                st.warning(f"⚠️ Could not load labels: {e}")
        
        # Create tabs for better organization
        tab1, tab2, tab3 = st.tabs(["📝 Setup & Labels", "📊 Analysis Results", "💾 Export"])
        
        with tab1:
            st.header("🧬 Experimental Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Cell Culture & Transfection")
                cells_seeded = st.text_input(
                    "Number of Cells Seeded (per well)*",
                    value="",
                    help="Required: e.g., 50,000",
                    key="cells_seeded_input"
                )
                hrs_to_transfection = st.text_input(
                    "Hours After Seeding to Transfection*",
                    value="",
                    help="Required: e.g., 24",
                    key="hrs_to_transfection_input"
                )
                biosensor_amount = st.text_input(
                    "Biosensor Transfected (ng per well)*",
                    value="",
                    help="Required: e.g., 100",
                    key="biosensor_amount_input"
                )
            
            with col2:
                st.subheader("Assay Timeline")
                gpcr_amount = st.text_input(
                    "GPCR Transfected (ng per well)*",
                    value="",
                    help="Required: e.g., 50",
                    key="gpcr_amount_input"
                )
                hrs_to_assay = st.text_input(
                    "Hours After Transfection to Assay*",
                    value="",
                    help="Required: e.g., 48",
                    key="hrs_to_assay_input"
                )
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📝 Methods Description")
                methods_text = st.text_area(
                    "Describe your experimental protocol:",
                    value="",
                    height=150,
                    help="Detail your experimental methods",
                    key="methods_text_input"
                )
            
            with col2:
                st.subheader("📌 Additional Notes")
                notes_text = st.text_area(
                    "Observations or highlights:",
                    value="",
                    height=150,
                    help="Optional: Any important notes",
                    key="notes_text_input"
                )
            
            st.markdown("---")
            st.header("🏷️ Well Labels")
            st.markdown("**Tip:** Wells with the same label will be treated as replicates")
            
            # Column headers
            cols = st.columns(13)
            cols[0].write("")  # Empty for row labels
            for col_idx in range(12):
                with cols[col_idx + 1]:
                    st.markdown(f"**{col_idx+1}**")
            
            # Label input grid
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
                            key=f"tab1_label_{well_name}",
                            label_visibility="collapsed"
                        )
                        labels_matrix[row_idx, col_idx] = label if label else ""
        
        # Process export data
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
        replicate_df = analyze_replicates(export_df)
        
        with tab2:
            st.header("📊 Analysis Results")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Total Wells Analyzed",
                    value=len(export_df)
                )
            
            with col2:
                active_wells = len(export_df[export_df['Fold Change'] > 2.0])
                st.metric(
                    label="Active Wells (FC > 2)",
                    value=active_wells,
                    delta=f"{active_wells/len(export_df)*100:.1f}%" if len(export_df) > 0 else "0%"
                )
            
            with col3:
                if replicate_df is not None:
                    st.metric(
                        label="Replicate Groups",
                        value=len(replicate_df)
                    )
                else:
                    st.metric(
                        label="Replicate Groups",
                        value=0
                    )
            
            st.markdown("---")
            
            if replicate_df is not None:
                st.subheader("🔬 Replicate Statistics")
                st.markdown("**Wells with identical labels are automatically grouped as replicates**")
                
                # Color code based on CV%
                def highlight_cv(row):
                    if row['CV%'] > 20:
                        return ['background-color: #ffcccc'] * len(row)
                    elif row['CV%'] > 10:
                        return ['background-color: #fff4cc'] * len(row)
                    else:
                        return ['background-color: #ccffcc'] * len(row)
                
                styled_df = replicate_df.style.apply(highlight_cv, axis=1)
                st.dataframe(styled_df, use_container_width=True)
                
                st.caption("🟢 CV < 10% (Excellent)  |  🟡 CV 10-20% (Acceptable)  |  🔴 CV > 20% (Poor - check for outliers)")
                st.markdown("---")
            
            st.subheader("📋 Individual Well Data")
            st.dataframe(export_df, use_container_width=True, height=400)
        
        with tab3:
            st.header("💾 Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
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
            
            with col2:
                st.subheader("📄 PDF Report")
                st.markdown("Comprehensive report with visualizations")
                
                # Check required fields
                required_fields_filled = all([cells_seeded, hrs_to_transfection, biosensor_amount, gpcr_amount, hrs_to_assay])
                
                if not required_fields_filled:
                    st.warning("⚠️ Please fill in all required experimental parameters in the Setup tab")
                    st.button("📥 Generate PDF Report", disabled=True, use_container_width=True)
                else:
                    if st.button("📥 Generate PDF Report", use_container_width=True, type="primary"):
                        with st.spinner("Generating comprehensive PDF report..."):
                            try:
                                kinetics_path = create_individual_well_kinetics(df_pre, df_post, labels_matrix, fc_matrix)
                                heatmap_path = create_fold_change_heatmap(fc_matrix, labels_matrix)
                                
                                fig_files_info = [
                                    ("Fold Change Heatmap", heatmap_path),
                                    ("Individual Well Kinetics (96-Well Format)", kinetics_path)
                                ]
                                
                                meta = {
                                    'experiment_name': experiment_name,
                                    'name_input': name_input,
                                    'methods_text': methods_text,
                                    'cells_seeded': cells_seeded,
                                    'hrs_to_transfection': hrs_to_transfection,
                                    'biosensor_amount': biosensor_amount,
                                    'gpcr_amount': gpcr_amount,
                                    'hrs_to_assay': hrs_to_assay,
                                    'notes_text': notes_text,
                                    'replicate_df': replicate_df
                                }
                                
                                pdf_bytes = generate_pdf_bytes(
                                    fig_files_info, labels_matrix, fc_matrix,
                                    export_df, df_pre, df_post, meta
                                )
                                
                                try:
                                    os.unlink(kinetics_path)
                                    os.unlink(heatmap_path)
                                except:
                                    pass
                                
                                if pdf_bytes:
                                    st.success("✅ PDF generated successfully!")
                                    st.download_button(
                                        label="📥 Download PDF Report",
                                        data=pdf_bytes,
                                        file_name=f"{experiment_name}_glosensor_report.pdf",
                                        mime="application/pdf",
                                        use_container_width=True
                                    )
                                else:
                                    st.error("Failed to generate PDF")
                            except Exception as e:
                                st.error(f"Error generating PDF: {e}")
                                import traceback
                                st.code(traceback.format_exc())
    
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
