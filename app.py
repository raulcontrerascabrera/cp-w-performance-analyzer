
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="CP-W′ Performance Analyzer", layout="wide")

@st.cache_data
def load_data(file):
    xl = pd.ExcelFile(file)
    trials = pd.read_excel(file, sheet_name="trials_CP")
    allout = pd.read_excel(file, sheet_name="three_min_allout")
    intervals = pd.read_excel(file, sheet_name="interval_templates")
    key = pd.read_excel(file, sheet_name="instructor_key")
    return trials, allout, intervals, key

st.title("📊 CP-W′ Performance Analyzer")

excel_file = "practica_potencia_critica_colab_datos.xlsx"

trials, allout, intervals, key = load_data(uploaded)

trials["work_J"] = trials["mean_power_W"] * trials["duration_s"]

results = []

    for athlete, df in trials.groupby("athlete_id"):
        X = df[["duration_s"]]
        y = df["work_J"]

        model = LinearRegression()
        model.fit(X, y)

        cp = model.coef_[0]
        wp = model.intercept_
        r2 = model.score(X, y)

        results.append([athlete, cp, wp, wp/1000, r2])

    summary = pd.DataFrame(
        results,
        columns=["Athlete","CP_W","Wprime_J","Wprime_kJ","R2"]
    )

    athlete = st.sidebar.selectbox(
        "Seleccionar atleta",
        summary["Athlete"].tolist()
    )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Dashboard",
    "Perfil",
    "3-min All-Out",
    "HIIT",
    "Interpretación",
    "Perfil Fisiológico",
    "Comparación"
])

    with tab1:

        st.subheader("Resumen")

        c1,c2,c3,c4 = st.columns(4)

        c1.metric("CP media", round(summary.CP_W.mean(),1))
        c2.metric("W′ medio (kJ)", round(summary.Wprime_kJ.mean(),1))
        c3.metric("Mayor CP", round(summary.CP_W.max(),1))
        c4.metric("Mayor W′", round(summary.Wprime_kJ.max(),1))

        st.dataframe(summary.round(2), use_container_width=True)

        fig = px.bar(summary, x="Athlete", y="CP_W")
        st.plotly_chart(fig, use_container_width=True)

    athlete_row = summary[summary.Athlete==athlete].iloc[0]
    athlete_df = trials[trials.athlete_id==athlete]

    with tab2:

        st.subheader(f"Perfil atleta {athlete}")

        X = athlete_df[["duration_s"]]
        y = athlete_df["work_J"]

        model = LinearRegression().fit(X,y)

        pred = model.predict(X)

        fig = go.Figure()
        fig.add_scatter(x=athlete_df["duration_s"], y=y,
                        mode="markers", name="Observado")
        fig.add_scatter(x=athlete_df["duration_s"], y=pred,
                        mode="lines", name="Modelo")
        st.plotly_chart(fig, use_container_width=True)

        cp = athlete_row.CP_W
        wp = athlete_row.Wprime_J

        t = np.linspace(60,1200,500)

        fig2 = go.Figure()
        fig2.add_scatter(
            x=t,
            y=cp + wp/t,
            mode="lines",
            name="Modelo"
        )
        fig2.add_scatter(
            x=athlete_df["duration_s"],
            y=athlete_df["mean_power_W"],
            mode="markers",
            name="Datos"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:

        CP_3min = allout.loc[allout["time_s"]>=155,"power_W"].mean()

        Wp_3min = (
            (allout["power_W"] - CP_3min)
            .clip(lower=0)
            .mul(5)
            .sum()
        )

        st.metric("CP 3-min", round(CP_3min,1))
        st.metric("W′ 3-min", round(Wp_3min/1000,2))

        fig = px.line(
            allout,
            x="time_s",
            y="power_W"
        )
        fig.add_hline(y=CP_3min)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:

        cp = athlete_row.CP_W
        wp = athlete_row.Wprime_J

        out = []

        for _,r in intervals.iterrows():

            gasto = (
                (r["work_power_%CP"]/100*cp)-cp
            ) * r["work_duration_s"]

            reps = wp/gasto

            out.append([
                r["session_id"],
                gasto,
                reps
            ])

        out = pd.DataFrame(
            out,
            columns=[
                "Sesion",
                "Gasto_rep_J",
                "Reps_hasta_fallo"
            ]
        )

        st.dataframe(out.round(2), use_container_width=True)

    with tab5:

        cp_pct = summary["CP_W"].rank(pct=True)[summary["Athlete"]==athlete].iloc[0]
        wp_pct = summary["Wprime_J"].rank(pct=True)[summary["Athlete"]==athlete].iloc[0]

        st.subheader("Interpretación fisiológica")

        if cp_pct > 0.75:
            cp_text = "CP elevada"
        elif cp_pct > 0.50:
            cp_text = "CP moderadamente elevada"
        else:
            cp_text = "CP moderada o baja"

        if wp_pct > 0.75:
            wp_text = "W′ elevada"
        elif wp_pct > 0.50:
            wp_text = "W′ moderada"
        else:
            wp_text = "W′ limitada"

        st.markdown(f"""
        ### Resumen

        El atleta presenta una **{cp_text}** y una **{wp_text}** dentro del grupo analizado.

        La CP obtenida fue de **{cp:.1f} W** y el W′ de **{athlete_row.Wprime_kJ:.1f} kJ**.

        Estos resultados sugieren un perfil fisiológico dependiente de la combinación entre capacidad aeróbica sostenible y tolerancia al trabajo realizado por encima de la potencia crítica.
        """)

    with tab6:

        st.subheader("Perfil fisiológico")

        cp_pct = summary["CP_W"].rank(pct=True)[summary["Athlete"]==athlete].iloc[0]
        wp_pct = summary["Wprime_J"].rank(pct=True)[summary["Athlete"]==athlete].iloc[0]

        cp_norm = cp_pct * 100
        wp_norm = wp_pct * 100

        aerobic_index = cp_norm
        severe_index = wp_norm

        fatigue_index = (
        athlete_df["mean_power_W"].max()
        /
        athlete_df["mean_power_W"].min()
        ) * 50

        fatigue_index = min(fatigue_index,100)

        global_score = (
        aerobic_index*0.4 +
        severe_index*0.4 +
        fatigue_index*0.2
        )

        c1,c2,c3,c4 = st.columns(4)

        c1.metric("Aerobic Index", f"{aerobic_index:.1f}")
        c2.metric("Severe Domain Index", f"{severe_index:.1f}")
        c3.metric("Fatigue Resistance", f"{fatigue_index:.1f}")
        c4.metric("Global Score", f"{global_score:.1f}")

        fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=global_score,
        title={'text': "Global Performance Score"},
        gauge={
            'axis': {'range': [0,100]},
            'bar': {'thickness':0.4}
        }
        ))

        st.plotly_chart(fig, use_container_width=True)

        if global_score >= 75:
         st.success("Perfil fisiológico: Excelente")
        elif global_score >= 50:
         st.info("Perfil fisiológico: Alto")
        elif global_score >= 25:
         st.warning("Perfil fisiológico: Medio")
        else:
         st.error("Perfil fisiológico: Bajo")

        st.markdown("---")

        if aerobic_index > severe_index + 15:
         st.markdown("### Perfil predominante: Aeróbico resistente")
        elif severe_index > aerobic_index + 15:
         st.markdown("### Perfil predominante: Especialista dominio severo")
        else:
         st.markdown("### Perfil predominante: Mixto / equilibrado")


    with tab7:

        st.subheader("Comparación entre atletas")

        radar = go.Figure()

        for _, r in summary.iterrows():

            radar.add_trace(
                go.Scatterpolar(
                    r=[
                        r["CP_W"],
                        r["Wprime_kJ"],
                        r["R2"]*100
                    ],
                    theta=[
                        "CP",
                        "W′",
                        "R²"
                    ],
                    fill="toself",
                    name=r["Athlete"]
                )
            )
    
        radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True
                )
            ),
            showlegend=True
        )
    
        st.plotly_chart(
            radar,
            use_container_width=True
        )
    
        leaderboard = summary.copy()
    
        leaderboard["Score"] = (
            leaderboard["CP_W"].rank(pct=True)*50 +
            leaderboard["Wprime_kJ"].rank(pct=True)*50
        )
    
        leaderboard = leaderboard.sort_values(
            "Score",
            ascending=False
        )
    
        st.subheader("Leaderboard")
    
        st.dataframe(
            leaderboard.round(2),
            use_container_width=True
        )
