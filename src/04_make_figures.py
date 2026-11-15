"""
04_make_figures.py
Generates high-resolution academic and project figures from Spark processed tables.
Outputs are saved into figures/ for use in reports, presentations, and the Streamlit dashboard.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from common import SUMMARIES_DIR, METRICS_DIR, PREDICTIONS_DIR, ALERTS_DIR, FIGURES_DIR

# Set global matplotlib style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

def make_all_figures():
    print("=" * 60)
    print("STAGE 4: Generating Publication-Quality Figures")
    print("=" * 60)

    # 1. Hourly Congestion Profile (Dual Axis)
    hourly_path = os.path.join(SUMMARIES_DIR, "hourly_profile.csv")
    if os.path.exists(hourly_path):
        df_h = pd.read_csv(hourly_path)
        fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
        
        bars = ax1.bar(df_h['hour'], df_h['congestion_rate_pct'], color='#E67E22', width=0.6, alpha=0.85, label='Congestion Rate (%)')
        ax1.set_xlabel('Hour of Day (0–23)', fontsize=11, fontweight='bold', color='#1A2530')
        ax1.set_ylabel('Congestion Rate (%)', fontsize=11, fontweight='bold', color='#E67E22')
        ax1.set_ylim(0, 65)
        ax1.set_xticks(range(0, 24))
        ax1.grid(True, linestyle='--', alpha=0.5)

        ax2 = ax1.twinx()
        line = ax2.plot(df_h['hour'], df_h['avg_speed'], color='#20639B', linewidth=2.5, marker='o', label='Avg Speed (km/h)')
        ax2.set_ylabel('Average Speed (km/h)', fontsize=11, fontweight='bold', color='#20639B')
        ax2.set_ylim(15, 45)
        ax2.grid(False)

        # Highlight peaks
        ax1.axvspan(7.5, 10.5, color='#F39C12', alpha=0.15, label='Morning Peak (08–10)')
        ax1.axvspan(16.5, 19.5, color='#D35400', alpha=0.15, label='Evening Peak (17–19)')

        plt.title('Hourly Congestion Profile & Average Vehicle Speed (12 Junctions)', fontsize=13, fontweight='bold', pad=14, color='#1A2530')
        fig.tight_layout()
        out1 = os.path.join(FIGURES_DIR, "01_hourly_congestion_profile.png")
        plt.savefig(out1)
        plt.close()
        print(f"Generated: {out1}")

    # 2. Junction Ranking (Horizontal Bar Chart)
    junc_path = os.path.join(SUMMARIES_DIR, "junction_summary.csv")
    if os.path.exists(junc_path):
        df_j = pd.read_csv(junc_path).sort_values("congestion_rate_pct", ascending=True)
        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
        
        colors = ['#C0392B' if 'Central Bus' in name else '#20639B' for name in df_j['junction_name']]
        bars = ax.barh(df_j['junction_name'], df_j['congestion_rate_pct'], color=colors, height=0.65)
        
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.4, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', ha='left', fontsize=9.5, fontweight='bold', color='#1A2530')
            
        ax.set_xlabel('Congestion Rate (%)', fontsize=11, fontweight='bold', color='#1A2530')
        ax.set_xlim(0, 25)
        ax.set_title('Congestion Rate Across 12 Tiruchirappalli Junctions', fontsize=13, fontweight='bold', pad=14, color='#1A2530')
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        fig.tight_layout()
        out2 = os.path.join(FIGURES_DIR, "02_junction_ranking.png")
        plt.savefig(out2)
        plt.close()
        print(f"Generated: {out2}")

    # 3. Day of Week x Hour Heatmap
    dow_path = os.path.join(SUMMARIES_DIR, "dow_hour_heatmap.csv")
    if os.path.exists(dow_path):
        df_dow = pd.read_csv(dow_path)
        dow_pivot = df_dow.pivot(index="dow", columns="hour", values="congestion_rate_pct")
        dow_labels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
        dow_pivot.index = [dow_labels[i-1] for i in dow_pivot.index]

        fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
        cax = ax.imshow(dow_pivot.values, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        
        ax.set_xticks(range(24))
        ax.set_xticklabels(range(24))
        ax.set_yticks(range(len(dow_pivot.index)))
        ax.set_yticklabels(dow_pivot.index)
        ax.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
        ax.set_ylabel('Day of Week', fontsize=11, fontweight='bold')
        ax.set_title('Congestion Heatmap: Day of Week vs Hour of Day (%)', fontsize=13, fontweight='bold', pad=14)
        
        cbar = fig.colorbar(cax, orientation='vertical', pad=0.02)
        cbar.set_label('Congestion Rate (%)', fontsize=10, fontweight='bold')
        fig.tight_layout()
        out3 = os.path.join(FIGURES_DIR, "03_dow_hour_heatmap.png")
        plt.savefig(out3)
        plt.close()
        print(f"Generated: {out3}")

    # 4. Junction x Hour Heatmap
    jh_path = os.path.join(SUMMARIES_DIR, "junction_hour_heatmap.csv")
    if os.path.exists(jh_path):
        df_jh = pd.read_csv(jh_path)
        jh_pivot = df_jh.pivot(index="junction_name", columns="hour", values="congestion_rate_pct")
        
        fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
        cax = ax.imshow(jh_pivot.values, cmap='OrRd', aspect='auto', interpolation='nearest')
        
        ax.set_xticks(range(24))
        ax.set_xticklabels(range(24))
        ax.set_yticks(range(len(jh_pivot.index)))
        ax.set_yticklabels(jh_pivot.index, fontsize=9.5)
        ax.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
        ax.set_ylabel('Junction', fontsize=11, fontweight='bold')
        ax.set_title('Spatio-Temporal Congestion Matrix (12 Junctions x 24 Hours)', fontsize=13, fontweight='bold', pad=14)
        
        cbar = fig.colorbar(cax, orientation='vertical', pad=0.02)
        cbar.set_label('Congestion Rate (%)', fontsize=10, fontweight='bold')
        fig.tight_layout()
        out4 = os.path.join(FIGURES_DIR, "04_junction_hour_heatmap.png")
        plt.savefig(out4)
        plt.close()
        print(f"Generated: {out4}")

    # 5. Weather, Peak, Zone Impact (Multi-panel)
    w_path = os.path.join(SUMMARIES_DIR, "weather_impact.csv")
    p_path = os.path.join(SUMMARIES_DIR, "peak_vs_offpeak.csv")
    z_path = os.path.join(SUMMARIES_DIR, "zone_summary.csv")
    
    if os.path.exists(w_path) and os.path.exists(p_path) and os.path.exists(z_path):
        df_w = pd.read_csv(w_path)
        df_p = pd.read_csv(p_path)
        df_z = pd.read_csv(z_path)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)
        
        # Weather
        w_colors = ['#16A085', '#7F8C8D', '#2980B9']
        ax1.bar(df_w['weather'], df_w['congestion_rate_pct'], color=w_colors, width=0.55)
        for idx, row in df_w.iterrows():
            ax1.text(idx, row['congestion_rate_pct'] + 1, f"{row['congestion_rate_pct']:.1f}%", ha='center', fontweight='bold')
        ax1.set_ylim(0, 50)
        ax1.set_ylabel('Congestion Rate (%)', fontweight='bold')
        ax1.set_title('Impact of Weather', fontweight='bold')
        ax1.grid(True, axis='y', linestyle='--', alpha=0.5)

        # Peak vs Off-Peak
        labels_p = ['Off-Peak', 'Peak Hours']
        p_colors = ['#27AE60', '#C0392B']
        ax2.bar(labels_p, df_p['congestion_rate_pct'], color=p_colors, width=0.45)
        for idx, row in df_p.iterrows():
            ax2.text(idx, row['congestion_rate_pct'] + 1.5, f"{row['congestion_rate_pct']:.1f}%", ha='center', fontweight='bold')
        ax2.set_ylim(0, 75)
        ax2.set_ylabel('Congestion Rate (%)', fontweight='bold')
        ax2.set_title('Peak vs Off-Peak', fontweight='bold')
        ax2.grid(True, axis='y', linestyle='--', alpha=0.5)

        # Zone
        z_colors = ['#8E44AD', '#2980B9', '#D35400', '#F39C12', '#16A085', '#2C3E50', '#BDC3C7']
        ax3.barh(df_z['zone'], df_z['congestion_rate_pct'], color=z_colors[:len(df_z)], height=0.6)
        for idx, row in df_z.iterrows():
            ax3.text(row['congestion_rate_pct'] + 0.5, idx, f"{row['congestion_rate_pct']:.1f}%", va='center', fontweight='bold')
        ax3.set_xlim(0, 25)
        ax3.set_xlabel('Congestion Rate (%)', fontweight='bold')
        ax3.set_title('Congestion by Urban Zone', fontweight='bold')
        ax3.grid(True, axis='x', linestyle='--', alpha=0.5)

        fig.suptitle('Traffic Contextual Drivers: Weather, Commute Peaks & Urban Zones', fontsize=13, fontweight='bold', y=1.03)
        fig.tight_layout()
        out5 = os.path.join(FIGURES_DIR, "05_contextual_impacts.png")
        plt.savefig(out5)
        plt.close()
        print(f"Generated: {out5}")

    # 6. Daily Congestion Rate Timeline
    daily_path = os.path.join(SUMMARIES_DIR, "daily_trend.csv")
    if os.path.exists(daily_path):
        df_d = pd.read_csv(daily_path)
        fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)
        ax.plot(pd.to_datetime(df_d['date']), df_d['congestion_rate_pct'], color='#C0392B', linewidth=2.2, marker='.')
        ax.set_ylabel('Congestion Rate (%)', fontsize=11, fontweight='bold', color='#1A2530')
        ax.set_xlabel('Date (August – September 2026)', fontsize=11, fontweight='bold', color='#1A2530')
        ax.set_ylim(0, 28)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(rotation=35, ha='right')
        plt.title('Daily Congestion Rate Timeline (60 Days Feed)', fontsize=13, fontweight='bold', pad=14)
        fig.tight_layout()
        out6 = os.path.join(FIGURES_DIR, "06_daily_congestion_trend.png")
        plt.savefig(out6)
        plt.close()
        print(f"Generated: {out6}")

    print("Stage 4 figure generation complete.")

if __name__ == "__main__":
    make_all_figures()
