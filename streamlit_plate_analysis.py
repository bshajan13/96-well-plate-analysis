import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.graph_objects as go

st.set_page_config(page_title="Plate Reader Fold Change", layout="wide")
st.title("📊 Plate Reader Fold Change Dashboard")

# --- 96-Well Template Download ---
st.sidebar.header("Download 96-Well Name Template")
template_matrix = pd.DataFrame([["" for _ in range(12)] for _ in range(8)])
buffer = BytesIO()
template_matrix.to_excel(buffer, index=False, header=False)
st.sidebar.download_button(
    label="Download Empty 96-Well Template",
    data=buffer,
    file_name="96_well_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# --- Uploads ---
st.sidebar.header("Upload Excel File (Data)")
uploaded_file = st.sidebar.file_uploader("Upload Plate Reader Excel (.xlsx)", type=["xlsx"])

st.sidebar.header("Upload 96-Well Name Template (Optional)")
uploaded_template = st.sidebar.file_uploader("Upload Name Template (.xlsx)", type=["xlsx"])

def make_unique(headers):
    """Make duplicate headers unique by appending _1, _2, etc."""
    counts = {}
    new_headers = []
    for h in headers:
        h_str = str(h)
        if h_str not in counts:
            counts[h_str] = 0
            new_headers.append(h_str)
        else:
            counts[h_str] += 1
            new_headers.append(f"{h_str}_{counts[h_str]}")
    return new_headers

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file, header=None)

        # -----------------------------
        # Plate 1 extraction (C60–CU68)
        # -----------------------------
        plate1_headers = df.loc[59, 2:100].dropna().to_numpy()  # row 60 = index 59
        if len(plate1_headers) == 0:
            st.error("❌ Could not find Plate 1 headers. Check Excel format.")
            st.stop()
        plate1_values = df.loc[60:67, 2:2+len(plate1_headers)].copy()
        plate1_values.columns = make_unique(plate1_headers)
        plate1_values.reset_index(drop=True, inplace=True)

        # -----------------------------
        # Plate 2 extraction (C118–CU126)
        # -----------------------------
        plate2_headers = df.loc[117, 2:100].dropna()  # row 118 = index 117
        if len(plate2_headers) == 0:
            st.error("❌ Could not find Plate 2 headers. Check Excel format.")
            st.stop()
        plate2_values = df.loc[118:125, 2:2+len(plate2_headers)].copy()
        plate2_values.columns = make_unique(plate2_headers)
        plate2_values.reset_index(drop=True, inplace=True)

        # -----------------------------
        # Convert to numeric
        # -----------------------------
        plate1_num = plate1_values.apply(pd.to_numeric, errors='coerce')
        plate2_num = plate2_values.apply(pd.to_numeric, errors='coerce')

        # -----------------------------
        # Full plate previews
        # -----------------------------
        st.subheader("🧫 Plate 1 Preview (Full)")
        st.dataframe(plate1_num, use_container_width=True)

        st.subheader("🧫 Plate 2 Preview (Full)")
        st.dataframe(plate2_num, use_container_width=True)

        # -----------------------------
        # Last 3 readings + averages
        # -----------------------------
        numeric_plate1 = plate1_num.iloc[:, 1:]
        numeric_plate2 = plate2_num.iloc[:, 1:]

        last3_plate1 = numeric_plate1.iloc[-3:, :]
        last3_plate2 = numeric_plate2.iloc[-3:, :]

        avg_plate1 = last3_plate1.mean(axis=0).round(1)
        avg_plate2 = last3_plate2.mean(axis=0).round(1)

        last3_plate1_display = pd.concat([last3_plate1, avg_plate1.to_frame().T.rename(index={0: 'Average'})])
        last3_plate2_display = pd.concat([last3_plate2, avg_plate2.to_frame().T.rename(index={0: 'Average'})])

        # -----------------------------
        # Expander: Plate 1 Last 3 Readings + Average
        # -----------------------------
        with st.expander("🧫 Plate 1: Last 3 Readings + Average", expanded=False):
            st.dataframe(last3_plate1_display, use_container_width=True)

        # -----------------------------
        # Expander: Plate 2 Last 3 Readings + Average
        # -----------------------------
        with st.expander("🧫 Plate 2: Last 3 Readings + Average", expanded=False):
            st.dataframe(last3_plate2_display, use_container_width=True)

        # -----------------------------
        # Expander: Fold Change Table
        # -----------------------------
        fold_change_table = (avg_plate2 / avg_plate1).round(2)
        with st.expander("📊 Fold Change Table", expanded=False):
            st.dataframe(fold_change_table.to_frame().T, use_container_width=True)

        # -----------------------------
        # Heatmap
        # -----------------------------
        num_rows = 8
        num_cols = 12
        row_labels = ['A','B','C','D','E','F','G','H']

        fc_values = fold_change_table.values.reshape((num_rows, num_cols))

        # Names only from uploaded template
        if uploaded_template:
            template_df = pd.read_excel(uploaded_template, header=None)
            if template_df.shape[0] >= 8 and template_df.shape[1] >= 12:
                name_matrix = template_df.iloc[:8, :12].astype(str).values
            else:
                st.warning("Template not 8x12, leaving names blank.")
                name_matrix = np.full((8, 12), "", dtype=str)
        else:
            name_matrix = np.full((8, 12), "", dtype=str)

        # Heatmap annotations
        annotations = []
        for i in range(num_rows):
            for j in range(num_cols):
                val = fc_values[i, j]
                drug_name = name_matrix[i, j]
                text = f"{drug_name}<br>{val}" if drug_name else f"{val}"
                annotations.append(dict(x=j, y=i, text=text, showarrow=False, font=dict(color="black", size=12)))

        fig = go.Figure(data=go.Heatmap(
            z=np.clip(fc_values, 0.5, 10),
            x=[f"Col {i+1}" for i in range(num_cols)],
            y=row_labels,
            colorscale=[[0.0, 'red'], [0.5, 'white'], [1.0, 'green']],
            zmid=1,
            showscale=True,
            hoverinfo="skip"
        ))
        fig.update_layout(
            annotations=annotations,
            xaxis=dict(side="top"),
            yaxis=dict(autorange="reversed"),
            width=1000,
            height=500,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        st.subheader("🟦 96-Well Plate Fold Change Heatmap")
        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # Column-wise Excel export
        # -----------------------------
        wells = []
        drugs = []
        fc_list = []

        for j in range(num_cols):         # column first
            for i in range(num_rows):     # then row
                wells.append(f"{row_labels[i]}{j+1}")
                drugs.append(name_matrix[i, j])
                fc_list.append(fc_values[i, j])

        export_df = pd.DataFrame({
            "Well": wells,
            "Drug": drugs,
            "Fold Change": fc_list
        })

        st.subheader("📋 Fold Change per Well (Excel Export)")
        st.dataframe(export_df, use_container_width=True)

        excel_buffer = BytesIO()
        export_df.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)

        st.download_button(
            label="📥 Download Excel with Fold Change per Well",
            data=excel_buffer,
            file_name="96_well_fold_change.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

else:
    st.info("👆 Upload your Excel file to begin.")
