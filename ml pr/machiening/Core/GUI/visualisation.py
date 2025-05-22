import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import pandas as pd
import numpy as np
import webbrowser
import os

class Visualise(ctk.CTkFrame):
    def __init__(self, parent, df, report_path=r"../EDA,ML/patientreport.html"):
        super().__init__(parent, fg_color="#f0f4f8")
        self.pack(fill="both", expand=True, padx=15, pady=15)
        self.df = df
        self.report_path = report_path
        self.current_canvas = None  # Track current plot canvas
        self.current_widget = None  # Track current widget (for report button or error label)
        self.setup_ui()

    def setup_ui(self):
        # Frame 1: Buttons (Left)
        self.button_frame = ctk.CTkFrame(self, fg_color="#ffffff", width=300, corner_radius=10)
        self.button_frame.pack(side="left", fill="y", padx=(10, 5), pady=10)
        self.button_frame.pack_propagate(False)

        # Frame 2: Plots (Right)
        self.plot_frame = ctk.CTkFrame(self, fg_color="#ffffff", corner_radius=10)
        self.plot_frame.pack(side="right", fill="both", expand=True, padx=(5, 10), pady=10)

        # Button frame: Add buttons for each visualization
        ctk.CTkLabel(self.button_frame, text="Visualization Selection", font=("Arial", 18, "bold")).pack(pady=(10, 20))

        ctk.CTkButton(self.button_frame, text="Show Histogram", command=self.show_histogram).pack(fill="x", padx=20, pady=10)
        ctk.CTkButton(self.button_frame, text="Show Heatmap", command=self.show_heatmap).pack(fill="x", padx=20, pady=10)
        ctk.CTkButton(self.button_frame, text="Show Pairplot", command=self.show_pairplot).pack(fill="x", padx=20, pady=10)
        ctk.CTkButton(self.button_frame, text="Show Boxplot", command=self.show_boxplot).pack(fill="x", padx=20, pady=10)
        ctk.CTkButton(self.button_frame, text="Show Scatter", command=self.show_scatter).pack(fill="x", padx=20, pady=10)
        ctk.CTkButton(self.button_frame, text="Show Barchart", command=self.show_barchart).pack(fill="x", padx=20, pady=10)
        ctk.CTkButton(self.button_frame, text="View Report", command=self.show_report).pack(fill="x", padx=20, pady=10)

        # Plot frame: Title and initial message
        ctk.CTkLabel(self.plot_frame, text="Visualization Page", font=("Arial", 20, "bold")).pack(pady=(10, 5))
        ctk.CTkLabel(self.plot_frame, text="Select a visualization from the left to view plots", font=("Arial", 14), text_color="#64748B").pack(pady=(0, 20))

    def clear_plot_frame(self):
        # Clear existing plot or widget
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
            self.current_canvas = None
        if self.current_widget:
            self.current_widget.destroy()
            self.current_widget = None

    def show_histogram(self):
        self.clear_plot_frame()
        try:
            numeric_cols = self.df.select_dtypes(include='number').columns
            if not numeric_cols.empty:
                column = numeric_cols[0]
                fig, ax = plt.subplots(figsize=(8, 5))
                self.df['BMI'].hist(ax=ax, bins=20, color="#1976D2")
                ax.set_title(f"Histogram of {"BMI"}")
                ax.set_xlabel("BMI")
                ax.set_ylabel("Frequency")
                self.current_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
                self.current_canvas.draw()
                self.current_canvas.get_tk_widget().pack(fill="both", expand=True)
                plt.close(fig)
            else:
                self.current_widget = ctk.CTkLabel(self.plot_frame, text="No numeric columns available for histogram", text_color="red", font=("Arial", 14))
                self.current_widget.pack(expand=True)
        except Exception as e:
            self.current_widget = ctk.CTkLabel(self.plot_frame, text=f"Error generating histogram:\n{str(e)}", text_color="red", font=("Arial", 14))
            self.current_widget.pack(expand=True)

    def show_heatmap(self):
        self.clear_plot_frame()
        try:
            numeric_cols = self.df.select_dtypes(include='number').columns
            if not numeric_cols.empty:
                corr_matrix = self.df[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", ax=ax)
                ax.set_title("Correlation Heatmap")
                self.current_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
                self.current_canvas.draw()
                self.current_canvas.get_tk_widget().pack(fill="both", expand=True)
                plt.close(fig)
            else:
                self.current_widget = ctk.CTkLabel(self.plot_frame, text="No numeric columns available for heatmap", text_color="red", font=("Arial", 14))
                self.current_widget.pack(expand=True)
        except Exception as e:
            self.current_widget = ctk.CTkLabel(self.plot_frame, text=f"Error generating heatmap:\n{str(e)}", text_color="red", font=("Arial", 14))
            self.current_widget.pack(expand=True)

    def show_pairplot(self):
        self.clear_plot_frame()
        try:
            numeric_cols = self.df.select_dtypes(include='number').columns
            if len(numeric_cols) >= 2:
                selected_cols = numeric_cols[:4]
                pair_plot = sns.pairplot(self.df[selected_cols], diag_kind="hist")
                fig = pair_plot.figure
                fig.set_size_inches(8, 5)
                self.current_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
                self.current_canvas.draw()
                self.current_canvas.get_tk_widget().pack(fill="both", expand=True)
                plt.close(fig)
            else:
                self.current_widget = ctk.CTkLabel(self.plot_frame, text="Need at least 2 numeric columns for pairplot", text_color="red", font=("Arial", 14))
                self.current_widget.pack(expand=True)
        except Exception as e:
            self.current_widget = ctk.CTkLabel(self.plot_frame, text=f"Error generating pairplot:\n{str(e)}", text_color="red", font=("Arial", 14))
            self.current_widget.pack(expand=True)

    def show_boxplot(self):
        self.clear_plot_frame()
        try:
            numeric_cols = self.df.select_dtypes(include='number').columns
            if not numeric_cols.empty:
                fig, ax = plt.subplots(figsize=(8, 5))
                self.df[numeric_cols].boxplot(ax=ax)
                ax.set_title("Boxplot of Numeric Columns")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                self.current_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
                self.current_canvas.draw()
                self.current_canvas.get_tk_widget().pack(fill="both", expand=True)
                plt.close(fig)
            else:
                self.current_widget = ctk.CTkLabel(self.plot_frame, text="No numeric columns available for boxplot", text_color="red", font=("Arial", 14))
                self.current_widget.pack(expand=True)
        except Exception as e:
            self.current_widget = ctk.CTkLabel(self.plot_frame, text=f"Error generating boxplot:\n{str(e)}", text_color="red", font=("Arial", 14))
            self.current_widget.pack(expand=True)

    def show_scatter(self):
        self.clear_plot_frame()
        try:
            numeric_cols = self.df.select_dtypes(include='number').columns
            if len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[:2]
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(data=self.df, x=x_col, y=y_col, ax=ax)
                ax.set_title(f"Scatter Plot of {x_col} vs {y_col}")
                self.current_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
                self.current_canvas.draw()
                self.current_canvas.get_tk_widget().pack(fill="both", expand=True)
                plt.close(fig)
            else:
                self.current_widget = ctk.CTkLabel(self.plot_frame, text="Need at least 2 numeric columns for scatter plot", text_color="red", font=("Arial", 14))
                self.current_widget.pack(expand=True)
        except Exception as e:
            self.current_widget = ctk.CTkLabel(self.plot_frame, text=f"Error generating scatter plot:\n{str(e)}", text_color="red", font=("Arial", 14))
            self.current_widget.pack(expand=True)

    def show_barchart(self):
        self.clear_plot_frame()
        try:
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
            if not cat_cols.empty:
                column = cat_cols[0]
                value_counts = self.df[column].value_counts()
                fig, ax = plt.subplots(figsize=(8, 5))
                value_counts.plot(kind='bar', ax=ax, color="#1976D2")
                ax.set_title(f"Bar Chart of {column}")
                ax.set_xlabel(column)
                ax.set_ylabel("Count")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                self.current_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
                self.current_canvas.draw()
                self.current_canvas.get_tk_widget().pack(fill="both", expand=True)
                plt.close(fig)
            else:
                self.current_widget = ctk.CTkLabel(self.plot_frame, text="No categorical columns available for bar chart", text_color="red", font=("Arial", 14))
                self.current_widget.pack(expand=True)
        except Exception as e:
            self.current_widget = ctk.CTkLabel(self.plot_frame, text=f"Error generating bar chart:\n{str(e)}", text_color="red", font=("Arial", 14))
            self.current_widget.pack(expand=True)

    def show_report(self):
        self.clear_plot_frame()
        try:
            if not os.path.exists(self.report_path):
                raise FileNotFoundError(f"Report file not found: {self.report_path}")
            self.current_widget = ctk.CTkButton(self.plot_frame, text="View Report", command=lambda: webbrowser.open(self.report_path))
            self.current_widget.pack(expand=True, pady=20)
        except Exception as e:
            self.current_widget = ctk.CTkLabel(self.plot_frame, text=f"Error setting up report button:\n{str(e)}", text_color="red", font=("Arial", 14))
            self.current_widget.pack(expand=True)
