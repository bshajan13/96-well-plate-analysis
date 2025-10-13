import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.graph_objects as go

st.set_page_config(page_title="Plate Reader Fold Change Dashboard", layout="wide")
st.title("📊 Plate Reader Fold Change Dashboard")

# --- 96-Well Setup ---
rows = list("ABCDEFGH")
cols = [str(i) for i in range(1, 13)]
well_names = [f"{r}{c}" for r in rows for c in cols]

# --- Sidebar Instructions ---
st.sidebar.header("📘 Instructions")
st.sidebar.write("""
1. Download the Excel template — it has three sheets:
   - **Labels** → for sample/compound names
   - **Pre (Plate 1)**
   - **Post (Plate 2)**
2. Fill in your data (Labels sheet empty by default).
3. Upload the completed file here.
4. The app will calculate fold change and generate:
   - Bar chart per well (colored by fold change, dotted line at y=1)
   - Heatmap with well IDs + Labels
""")

# --- Template Download ---
template_labels = pd.DataFrame(columns=cols, index=rows)  # 96-well layout
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

# --- Helper function ---
def process_sheet(xls, sheet_name):
    """Read one sheet and compute average of last 3 readings (excluding Time column)"""
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

        last3 = df.iloc[-3:, 1:]  # exclude Time column
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
            st.error("❌ Missing required sheets: 'Labels', 'Pre (Plate 1)', 'Post (Plate 2)'")
            st.stop()

        # --- Robust Labels Reading ---
        labels_df = pd.read_excel(xls, sheet_name="Labels", header=0)
        while labels_df.shape[0] < 8:
            labels_df = pd.concat([labels_df, pd.DataFrame([[""]*labels_df.shape[1]])], ignore_index=True)
        while labels_df.shape[1] < 12:
            labels_df[f"Extra_{labels_df.shape[1]+1}"] = ""
        labels_matrix = labels_df.iloc[:8, :12].fillna("").astype(str).values

        # --- Process Pre/Post Sheets ---
        pre_avg_vals, pre_avg_df = process_sheet(xls, "Pre (Plate 1)")
        post_avg_vals, post_avg_df = process_sheet(xls, "Post (Plate 2)")

        if pre_avg_vals is not None and post_avg_vals is not None:
            # --- Fold Change ---
            fold_change = (post_avg_vals / pre_avg_vals).replace([np.inf, -np.inf], np.nan).round(2)
            st.subheader("📈 Fold Change (Post ÷ Pre)")
            st.dataframe(fold_change.to_frame().T, use_container_width=True)

            # --- Flattened Data for Bar Chart ---
            fc_values = fold_change.values.reshape((8, 12))
            wells = [f"{r}{c}" for r in rows for c in cols]
            labels_flat = labels_matrix.flatten()
            fc_flat = fc_values.flatten()
            df_wells = pd.DataFrame({"Well": wells, "Name": labels_flat, "Fold Change": fc_flat})

            # --- Diverging Color Scale ---
            colorscale = [[0.0, 'red'], [0.5, 'white'], [1.0, 'green']]
            cmin, cmax = 0.1, 10

            # --- Bar Chart with dotted line at y=1 ---
            fig_wells = go.Figure(data=go.Bar(
                x=df_wells["Well"],
                y=df_wells["Fold Change"],
                text=df_wells["Fold Change"],
                marker=dict(
                    color=df_wells["Fold Change"],
                    colorscale=colorscale,
                    cmin=cmin,
                    cmax=cmax
                ),
                hovertext=df_wells["Name"],
                hoverinfo="x+y+text"
            ))
            fig_wells.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig_wells.update_layout(
                title="📊 Fold Change per Individual Well",
                xaxis_title="Well",
                yaxis_title="Fold Change"
            )
            fig_wells.add_hline(y=1, line=dict(color='black', dash='dot', width=0.8))
            st.plotly_chart(fig_wells, use_container_width=True)

            # --- Heatmap ---
            annotations = []
            for i in range(8):
                for j in range(12):
                    val = fc_values[i, j]
                    label = labels_matrix[i, j]
                    well = f"{rows[i]}{j+1}"
                    text = f"{well}: {label}<br>{val}" if label.strip() else f"{well}<br>{val}"
                    annotations.append(dict(
                        x=j, y=i, text=text, showarrow=False, font=dict(color="black", size=12)
                    ))

            fig_heatmap = go.Figure(data=go.Heatmap(
                z=np.clip(fc_values, 0.1, 10),
                x=[f"Col {i+1}" for i in range(12)],
                y=rows,
                colorscale=colorscale,
                zmid=1,
                showscale=True,
                hoverinfo="skip"
            ))
            fig_heatmap.update_layout(
                annotations=annotations,
                xaxis=dict(side="top"),
                yaxis=dict(autorange="reversed"),
                width=1000,
                height=500,
                margin=dict(l=50, r=50, t=50, b=50)
            )

            st.subheader("🟩 Fold Change Heatmap")
            st.plotly_chart(fig_heatmap, use_container_width=True)

            # --- Export Fold Change per Well ---
            export_df = pd.DataFrame({"Well": wells, "Name": labels_flat, "Fold Change": fc_flat})
            excel_buffer = BytesIO()
            export_df.to_excel(excel_buffer, index=False)
            excel_buffer.seek(0)
            st.download_button(
                label="📥 Download Fold Change per Well",
                data=excel_buffer,
                file_name="fold_change_per_well.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        else:
            st.info("👆 Please fill both Pre/Post sheets and re-upload to generate results.")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

else:
    st.info("👆 Download the Excel template, fill Labels + Pre/Post sheets, then upload it here.")
