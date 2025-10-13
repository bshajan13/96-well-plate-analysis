import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.graph_objects as go
from datetime import datetime

from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph,
    Spacer, Image, PageBreak
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# --- Streamlit Setup ---
st.set_page_config(page_title="Plate Reader Fold Change Dashboard", layout="wide")
st.title("📊 Plate Reader Fold Change Dashboard")

# --- Sidebar Instructions ---
st.sidebar.header("📘 Instructions")
st.sidebar.write("""
1. Download the Excel template — it has three sheets:
   - **Labels** → for sample/compound names  
   - **Pre (Plate 1)**  
   - **Post (Plate 2)**
2. Fill in your data (Labels sheet empty by default).  
3. Upload the completed file here.  
4. Enter your **Name** and **Experiment Name**.  
5. The app will calculate fold change, generate charts, and a PDF report.
""")

# --- Required Inputs ---
name_input = st.sidebar.text_input("Name:")
experiment_name = st.sidebar.text_input("Experiment Name:")
if not name_input.strip() or not experiment_name.strip():
    st.warning("⚠️ Please enter both Name and Experiment Name to continue.")
    st.stop()

# --- 96-Well Setup ---
rows = list("ABCDEFGH")
cols = [str(i) for i in range(1, 13)]
well_names = [f"{r}{c}" for r in rows for c in cols]

# --- Template Download ---
template_labels = pd.DataFrame(columns=cols, index=rows)
template_prepost = pd.DataFrame(columns=["Time (min)"] + well_names)

buffer = BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    template_labels.to_excel(writer, index=False, header=True, sheet_name="Labels")
    template_prepost.to_excel(writer, index=False, sheet_name="Pre (Plate 1)")
    template_prepost.to_excel(writer, index=False, sheet_name="Post (Plate 2)")
buffer.seek(0)

st.sidebar.download_button(
    label="📥 Download Template with Labels",
    data=buffer,
    file_name="plate_reader_template_with_labels.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# --- Upload File ---
uploaded_file = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

# --- Helper Function ---
def process_sheet(xls, sheet_name):
    try:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        if "Time (min)" not in df.columns:
            st.error(f"❌ {sheet_name}: Missing 'Time (min)' column.")
            return None, None

        st.subheader(f"🧫 {sheet_name} — Raw Data")
        st.dataframe(df, use_container_width=True)

        if len(df) < 3:
            st.warning(f"{sheet_name}: Not enough timepoints (need ≥3).")
            return None, None

        last3 = df.iloc[-3:, 1:]
        avg_vals = last3.mean(axis=0).round(2)
        avg_df = avg_vals.to_frame().T

        st.subheader(f"📊 {sheet_name} — Average of Last 3 Readings")
        st.dataframe(avg_df, use_container_width=True)

        return avg_vals, avg_df
    except Exception as e:
        st.error(f"❌ Error reading {sheet_name}: {e}")
        return None, None


# --- Main Processing ---
if uploaded_file:
    try:
        xls = pd.ExcelFile(uploaded_file)
        required_sheets = ["Labels", "Pre (Plate 1)", "Post (Plate 2)"]
        if not all(r in xls.sheet_names for r in required_sheets):
            st.error("❌ Missing required sheets")
            st.stop()

        # --- Labels ---
        labels_df = pd.read_excel(xls, sheet_name="Labels", header=0)
        while labels_df.shape[0] < 8:
            labels_df = pd.concat([labels_df, pd.DataFrame([[""] * labels_df.shape[1]])], ignore_index=True)
        while labels_df.shape[1] < 12:
            labels_df[f"Extra_{labels_df.shape[1] + 1}"] = ""

        labels_matrix = labels_df.iloc[:8, :12].fillna("").astype(str).values
        labels_flat = labels_matrix.flatten()
        wells = [f"{r}{c}" for r in rows for c in cols]

        # --- Process Pre/Post ---
        pre_avg_vals, pre_avg_df = process_sheet(xls, "Pre (Plate 1)")
        post_avg_vals, post_avg_df = process_sheet(xls, "Post (Plate 2)")

        if pre_avg_vals is not None and post_avg_vals is not None:
            fold_change = (post_avg_vals / pre_avg_vals).replace([np.inf, -np.inf], np.nan).round(2)
            fc_values = fold_change.values.reshape((8, 12))
            fc_flat = fc_values.flatten()
            export_df = pd.DataFrame({"Well": wells, "Name": labels_flat, "Fold Change": fc_flat})

            # --- Charts ---
            colorscale = [[0.0, 'red'], [0.5, 'white'], [1.0, 'green']]
            cmin, cmax = 0.1, 10

            # Bar chart
            fig_wells = go.Figure(data=go.Bar(
                x=export_df["Well"],
                y=export_df["Fold Change"],
                text=export_df["Fold Change"],
                marker=dict(color=export_df["Fold Change"], colorscale=colorscale, cmin=cmin, cmax=cmax),
                hovertext=export_df["Name"],
                hoverinfo="x+y+text"
            ))
            fig_wells.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig_wells.update_layout(width=1400)
            fig_wells.add_hline(y=1, line=dict(color='black', dash='dot', width=0.8))
            st.plotly_chart(fig_wells, use_container_width=True)

            # Heatmap
            annotations = []
            for i in range(8):
                for j in range(12):
                    val = fc_values[i, j]
                    label = labels_matrix[i, j]
                    well = f"{rows[i]}{j + 1}"
                    text = f"{well}: {label}<br>{val}" if label.strip() else f"{well}<br>{val}"
                    annotations.append(dict(x=j, y=i, text=text, showarrow=False, font=dict(color="black", size=12)))

            fig_heatmap = go.Figure(data=go.Heatmap(
                z=np.clip(fc_values, 0.1, 10),
                x=[f"Col {i + 1}" for i in range(12)],
                y=rows,
                colorscale=colorscale,
                zmid=1,
                showscale=True,
                hoverinfo="skip"
            ))
            fig_heatmap.update_layout(
                annotations=annotations,
                xaxis=dict(side="top", tickangle=-45),
                yaxis=dict(autorange="reversed"),
                width=1400,
                height=500
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

            # --- PDF Report ---
            pdf_buffer = BytesIO()
            doc = SimpleDocTemplate(
                pdf_buffer, pagesize=landscape(A4),
                leftMargin=15*mm, rightMargin=15*mm,
                topMargin=20*mm, bottomMargin=15*mm
            )

            elements = []
            styles = getSampleStyleSheet()
            center_style = ParagraphStyle(name='center', alignment=TA_CENTER, fontSize=14, spaceAfter=10)
            left_style = ParagraphStyle(name='left', alignment=TA_LEFT, fontSize=10)
            right_style = ParagraphStyle(name='right', alignment=TA_RIGHT, fontSize=10)

            # --- Page 1: Labels ---
            header_table = Table([
                [
                    Paragraph(f"Date: {datetime.now().strftime('%d/%m/%Y')}", left_style),
                    Paragraph(f"{experiment_name}", center_style),
                    Paragraph(f"Name: {name_input}", right_style)
                ]
            ], colWidths=[80*mm, 110*mm, 80*mm])
            elements.append(header_table)
            elements.append(Spacer(1, 12))
            elements.append(Paragraph("Sample Names", styles['Title']))
            elements.append(Spacer(1, 5))

            plate_labels = [[f"{rows[i]}{j+1}\n{labels_matrix[i, j]}" for j in range(12)] for i in range(8)]
            t_labels = Table(plate_labels, colWidths=20*mm, rowHeights=12*mm)
            t_labels.setStyle(TableStyle([
                ('GRID', (0,0), (-1,-1), 0.5, colors.black),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('FONTSIZE', (0,0), (-1,-1), 8)
            ]))
            elements.append(t_labels)
            elements.append(PageBreak())

            # --- Page 2: Fold Change Bar + Wells >2 ---
            elements.append(Paragraph("Fold Change per Well", styles['Title']))
            elements.append(Spacer(1, 5))
            bar_img = Image(BytesIO(fig_wells.to_image(format="png", scale=4)))
            bar_img.drawHeight = 120*mm
            bar_img.drawWidth = 270*mm
            elements.append(bar_img)
            elements.append(Spacer(1, 5))

            high_fc_wells = [(export_df.loc[i, "Well"], export_df.loc[i, "Name"], export_df.loc[i, "Fold Change"])
                             for i in range(len(export_df)) if export_df.loc[i, "Fold Change"] > 2]
            if high_fc_wells:
                text = "Wells with Fold Change > 2: " + ", ".join(
                    [f"{w[0]} ({w[1]}) = {w[2]}" for w in high_fc_wells]
                )
                elements.append(Paragraph(text, ParagraphStyle(name='highlight', fontSize=10, alignment=TA_LEFT, spaceAfter=10)))
            else:
                elements.append(Paragraph("No wells showed fold change > 2.",
                                          ParagraphStyle(name='highlight', fontSize=10, alignment=TA_LEFT, spaceAfter=10)))
            elements.append(PageBreak())

            # --- Page 3: Heatmap ---
            elements.append(Paragraph("Fold Change Heatmap", styles['Title']))
            elements.append(Spacer(1, 5))
            heat_img = Image(BytesIO(fig_heatmap.to_image(format="png", scale=4)))
            heat_img.drawHeight = 120*mm
            heat_img.drawWidth = 270*mm
            elements.append(heat_img)
            elements.append(PageBreak())

            # --- Page 4: Combined Table ---
            combined_data = []
            for i, well in enumerate(wells):
                label = labels_flat[i]
                pre_val = pre_avg_vals[well] if well in pre_avg_vals else ""
                post_val = post_avg_vals[well] if well in post_avg_vals else ""
                fc_val = fold_change[well] if well in fold_change else ""
                combined_data.append([well, label, pre_val, post_val, fc_val])

            headers = ["Well", "Label", "Pre Avg", "Post Avg", "Fold Change"]
            table_data = [headers] + combined_data

            t_combined = Table(
                table_data,
                colWidths=[20*mm, 60*mm, 30*mm, 30*mm, 30*mm],
                rowHeights=12
            )
            t_combined.setStyle(TableStyle([
                ('GRID', (0,0), (-1,-1), 0.5, colors.black),
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('FONTSIZE', (0,0), (-1,-1), 8)
            ]))
            elements.append(Paragraph("Pre Avg, Post Avg & Fold Change per Well", styles['Heading2']))
            elements.append(t_combined)
            elements.append(PageBreak())

            # --- Page 5: Raw Reads Side by Side (CLEAN HEADER VERSION) ---
            df_pre = pd.read_excel(xls, sheet_name="Pre (Plate 1)").set_index("Time (min)").T.reset_index()
            df_post = pd.read_excel(xls, sheet_name="Post (Plate 2)").set_index("Time (min)").T.reset_index()

            df_pre.rename(columns={"index": "Well"}, inplace=True)
            df_post.rename(columns={"index": "Well"}, inplace=True)

            df_pre.insert(1, "Label", labels_flat)
            df_post.insert(1, "Label", labels_flat)

            num_pre = len(df_pre.columns) - 2
            num_post = len(df_post.columns) - 2

            header_row1 = (
                ["", "", f"Pre (Plate 1)"] + [""] * (num_pre - 1)
                + [""] + ["", "", f"Post (Plate 2)"] + [""] * (num_post - 1)
            )
            header_row2 = (
                ["Well", "Label"] + list(df_pre.columns[2:])
                + [""] + list(df_post.columns[2:])
            )

            combined_data = []
            for i in range(len(df_pre)):
                row = list(df_pre.iloc[i]) + [""] + list(df_post.iloc[i, 2:])
                combined_data.append(row)

            col_widths = [15*mm, 30*mm] + [12*mm]*num_pre + [5*mm] + [12*mm]*num_post

            t_raw_side = Table(
                [header_row1, header_row2] + combined_data,
                colWidths=col_widths,
                rowHeights=12
            )
            t_raw_side.setStyle(TableStyle([
                ('SPAN', (2,0), (1+num_pre,0)),
                ('SPAN', (num_pre+2,0), (num_pre+1+num_post,0)),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('GRID', (0,1), (-1,-1), 0.5, colors.black),
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                ('BACKGROUND', (0,1), (-1,1), colors.whitesmoke),
                ('BACKGROUND', (num_pre+2,0), (num_pre+2,-1), colors.lightblue),  # separator column
                ('FONTSIZE', (0,0), (-1,-1), 8)
            ]))

            elements.append(Paragraph("Raw Reads: Pre (Plate 1) & Post (Plate 2) Side by Side", styles['Heading2']))
            elements.append(t_raw_side)
            elements.append(PageBreak())

            # --- Build PDF ---
            doc.build(elements)
            pdf_buffer.seek(0)

            st.download_button(
                label="📥 Download Full PDF Report",
                data=pdf_buffer,
                file_name="plate_reader_full_report.pdf",
                mime="application/pdf"
            )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
