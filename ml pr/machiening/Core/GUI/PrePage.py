import customtkinter as ctk
import webbrowser
from PIL import Image, ImageTk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import seaborn as sns
import os

class PrePage:
    def __init__(self, parent, df, app=None):
        self.parent = parent
        self.app = app
        # Use the app's DataFrame as the source of truth
        self.df = self.app.df if self.app and isinstance(self.app.df, pd.DataFrame) and not self.app.df.empty else pd.DataFrame()
        self.processed_df = self.df.copy()
        self.chart_frames = []
        self.image_chart_frames = []
        self.setup_ui()
        if not self.df.empty:
            self.populate_tabs()
        else:
            self.show_error("No valid dataset provided. Please upload a dataset in the Dataset tab.")

    def setup_ui(self):
        self.main_frame = ctk.CTkFrame(self.parent, fg_color="#f5f7fa")
        self.main_frame.pack(fill="both", expand=True, padx=15, pady=15)

        sidebar_frame = ctk.CTkScrollableFrame(
            self.main_frame, fg_color="#ffffff", width=250, corner_radius=10,
            border_width=1, border_color="#e2e8f0"
        )
        sidebar_frame.pack(side="left", fill="y", padx=(0, 10), pady=10)

        content_frame = ctk.CTkFrame(self.main_frame, fg_color="#ffffff", corner_radius=10, border_width=1, border_color="#e2e8f0")
        content_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(content_frame, text="Data Preprocessing", font=("Arial", 20, "bold"), text_color="#2d3748").pack(pady=(15, 5))
        ctk.CTkLabel(content_frame, text="Apply preprocessing steps and visualize results", font=("Arial", 14), text_color="#718096").pack(pady=(0, 20))

        self.tab_view = ctk.CTkTabview(content_frame)
        self.tab_view.pack(fill="both", expand=True, padx=10, pady=10)

        self.tabs = [
            "Info", "Null Values", "Null Percentage", "Mode Imputation", "Mean Imputation",
            "KNN Imputer", "Iterative Imputer", "Skewness", "Log Transformation",
            "Evaluate Log Effect", "Boxplot", "Label Encoding", "Histogram",
            "Standard Deviation", "Histogram After Std", "Z-Score",
            "Histogram After Z-Score", "Median Imputation", "Histogram After Median"
        ]
        for tab_name in self.tabs:
            tab = self.tab_view.add(tab_name)
            frame = ctk.CTkFrame(tab, fg_color="#f0f4f8", corner_radius=10)
            frame.pack(fill="both", expand=True, padx=10, pady=10)
            self.chart_frames.append(frame)
            btn_text = tab_name.replace("Histogram After ", "Hist After ")
            ctk.CTkButton(
                sidebar_frame, text=btn_text, font=("Arial", 14),
                fg_color="#4a90e2", hover_color="#357abd", text_color="white",
                command=lambda t=tab_name: self.tab_view.set(t)
            ).pack(fill="x", pady=5, padx=20)

        button_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        button_frame.pack(fill="x", pady=10)
        ctk.CTkButton(
            button_frame, text="Apply Preprocessing", font=("Arial", 14),
            fg_color="#4a90e2", hover_color="#357abd", text_color="white",
            command=self.apply_preprocessing
        )
        ctk.CTkButton(
            button_frame, text="Reset Data", font=("Arial", 14),
            fg_color="#e2e8f0", hover_color="#d1d9e6", text_color="#2d3748",
            command=self.reset_data
        ).pack(side="left", padx=10)

    def show_error(self, message):
        for frame in self.chart_frames:
            error_label = ctk.CTkLabel(frame, text=message, text_color="red", font=("Arial", 14))
            error_label.pack(expand=True)

    def populate_tabs(self):
        methods = [
            self.show_info, self.show_null_values, self.show_null_percentage,
            self.show_mode_imputation, self.show_mean_imputation, self.show_knn_imputation,
            self.show_iterative_imputation, self.show_skewness, self.show_log_transformation,
            self.evaluate_log_effect, self.show_boxplot, self.show_label_encoding,
            self.draw_histogram, self.show_standard_deviation, self.draw_histogram_after_std,
            self.show_z_score, self.draw_histogram_after_zscore, self.show_median_imputation,
            self.draw_histogram_after_median
        ]
        for frame, method in zip(self.chart_frames, methods):
            for widget in frame.winfo_children():
                widget.destroy()
            method(frame)

    def update_ui(self, df):
        """Update the UI with the dataset from the main app."""
        self.df = df if isinstance(df, pd.DataFrame) and not df.empty else pd.DataFrame()
        self.processed_df = self.df.copy()
        for frame in self.chart_frames:
            for widget in frame.winfo_children():
                widget.destroy()
        if not self.df.empty:
            self.populate_tabs()
        else:
            self.show_error("No valid dataset provided. Please upload a dataset in the Dataset tab.")

    def apply_preprocessing(self):
        try:
            if not self.app:
                raise ValueError("Main application reference not provided.")
            
            # Update the main app's DataFrame with the processed data
            self.app.df = self.processed_df.copy()
            
            # Update the Dataset page
            if self.app.dataset_page:
                self.app.dataset_page.update_ui(self.app.df, self.app.dataset_name)
            
            # Update the Visualization page by reinitializing it if it exists
            if self.app.visualization_page:
                self.app.clear_main_frame()
                self.app.visualization_page = Visualise(self.app.main_frame, self.app.df, self.app.report_path)
            
            # Update the ML Model page by reinitializing it if it exists
            if self.app.model_page:
                self.app.clear_main_frame()
                self.app.model_page = MLModelPage(self.app.main_frame, self.app.df)
                self.app.model_page.create_ui()
            
            # Refresh the current preprocessing page
            self.update_ui(self.app.df)
            
            messagebox.showinfo("Success", "Preprocessing applied and dataset updated across all tabs.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply preprocessing: {str(e)}")

    def reset_data(self):
        self.processed_df = self.df.copy()
        self.populate_tabs()
        messagebox.showinfo("Success", "Dataset reset to original state.")

    def show_info(self, frame):
        try:
            info_text = f"Dataset Info:\n\nShape: {self.df.shape}\n\nColumns: {list(self.df.columns)}\n\nDtypes:\n{self.df.dtypes}"
            text_box = ctk.CTkTextbox(frame, width=600, height=300, font=("Courier", 14))
            text_box.insert("0.0", info_text)
            text_box.configure(state="disabled")
            text_box.pack(expand=True, pady=10)
        except Exception as e:
            ctk.CTkLabel(frame, text=f"Error: {str(e)}", text_color="red", font=("Arial", 14)).pack(expand=True)

    def show_null_values(self, frame):
        try:
            null_counts = self.df.isnull().sum()
            null_text = "Null Values in Each Column:\n\n" + str(null_counts)
            text_box = ctk.CTkTextbox(frame, width=600, height=300, font=("Courier", 14))
            text_box.insert("0.0", null_text)
            text_box.configure(state="disabled")
            text_box.pack(expand=True, pady=10)
        except Exception as e:
            ctk.CTkLabel(frame, text=f"Error: {str(e)}", text_color="red", font=("Arial", 14)).pack(expand=True)

    def show_null_percentage(self, frame):
        try:
            null_percentage = (self.df.isnull().sum() / len(self.df)) * 100
            fig, ax = plt.subplots(figsize=(8, 5))
            null_percentage.plot(kind='bar', ax=ax, color="#4a90e2")
            ax.set_title("Percentage of Null Values per Column")
            ax.set_ylabel("Percentage (%)")
            ax.set_xlabel("Columns")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            plt.close(fig)
        except Exception as e:
            ctk.CTkLabel(frame, text=f"Error: {str(e)}", text_color="red", font=("Arial", 14)).pack(expand=True)

    def show_mode_imputation(self, frame):
        try:
            imputer = SimpleImputer(strategy="most_frequent")
            temp_df = pd.DataFrame(imputer.fit_transform(self.df), columns=self.df.columns)
            null_after = temp_df.isnull().sum()
            text = f"Null Values After Mode Imputation:\n\n{null_after}"
            text_box = ctk.CTkTextbox(frame, width=600, height=300, font=("Courier", 14))
            text_box.insert("0.0", text)
            text_box.configure(state="disabled")
            text_box.pack(expand=True, pady=10)
            ctk.CTkButton(
                frame, text="Apply Mode Imputation", font=("Arial", 14),
                fg_color="#4a90e2", hover_color="#357abd", text_color="white",
                command=lambda: self.apply_imputation(imputer)
            ).pack(pady=10)
        except Exception as e:
            ctk.CTkLabel(frame, text=f"Error: {str(e)}", text_color="red", font=("Arial", 14)).pack(expand=True)

    def show_mean_imputation(self, frame):
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if numeric_cols.empty:
                raise ValueError("No numeric columns available for mean imputation.")
            imputer = SimpleImputer(strategy="mean")
            temp_df = self.df.copy()
            temp_df[numeric_cols] = imputer.fit_transform(temp_df[numeric_cols])
            null_after = temp_df.isnull().sum()
            text = f"Null Values After Mean Imputation:\n\n{null_after}"
            text_box = ctk.CTkTextbox(frame, width=600, height=300, font=("Courier", 14))
            text_box.insert("0.0", text)
            text_box.configure(state="disabled")
            text_box.pack(expand=True, pady=10)
            ctk.CTkButton(
                frame, text="Apply Mean Imputation", font=("Arial", 14),
                fg_color="#4a90e2", hover_color="#357abd", text_color="white",
                command=lambda: self.apply_imputation(imputer, numeric_cols)
            ).pack(pady=10)
        except Exception as e:
            ctk.CTkLabel(frame, text=f"Error: {str(e)}", text_color="red", font=("Arial", 14)).pack(expand=True)

    def show_median_imputation(self, frame):
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if numeric_cols.empty:
                raise ValueError("No numeric columns available for median imputation.")
            imputer = SimpleImputer(strategy="median")
            temp_df = self.df.copy()
            temp_df[numeric_cols] = imputer.fit_transform(temp_df[numeric_cols])
            null_after = temp_df.isnull().sum()
            text = f"Null Values After Median Imputation:\n\n{null_after}"
            text_box = ctk.CTkTextbox(frame, width=600, height=300, font=("Courier", 14))
            text_box.insert("0.0", text)
            text_box.configure(state="disabled")
            text_box.pack(expand=True, pady=10)
            ctk.CTkButton(
                frame, text="Apply Median Imputation", font=("Arial", 14),
                fg_color="#4a90e2", hover_color="#357abd", text_color="white",
                command=lambda: self.apply_imputation(imputer, numeric_cols)
            ).pack(pady=10)
        except Exception as e:
            ctk.CTkLabel(frame, text=f"Error: {str(e)}", text_color="red", font=("Arial", 14)).pack(expand=True)

    def show_knn_imputation(self, frame):
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if numeric_cols.empty:
                raise ValueError("No numeric columns available for KNN imputation.")
            imputer = KNNImputer(n_neighbors=5)
            temp_df = self.df.copy()
            temp_df[numeric_cols] = imputer.fit_transform(temp_df[numeric_cols])
            null_after = temp_df.isnull().sum()
            text = f"Null Values After KNN Imputation:\n\n{null_after}"
            text_box = ctk.CTkTextbox(frame, width=600, height=300, font=("Courier", 14))
            text_box.insert("0.0", text)
            text_box.configure(state="disabled")
            text_box.pack(expand=True, pady=10)
            ctk.CTkButton(
                frame, text="Apply KNN Imputation", font=("Arial", 14),
                fg_color="#4a90e2", hover_color="#357abd", text_color="white",
                command=lambda: self.apply_imputation(imputer, numeric_cols)
            ).pack(pady=10)
        except Exception as e:
            ctk.CTkLabel(frame, text=f"Error: {str(e)}", text_color="red", font=("Arial", 14)).pack(expand=True)

    def show_iterative_imputation(self, frame):
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if numeric_cols.empty:
                raise ValueError("No numeric columns available for iterative imputation.")
            imputer = IterativeImputer(random_state=42)
            temp_df = self.df.copy()
            temp_df[numeric_cols] = imputer.fit_transform(temp_df[numeric_cols])
            null_after = temp_df.isnull().sum()
            text = f"Null Values After Iterative Imputation:\n\n{null_after}"
            text_box = ctk.CTkTextbox(frame, width=600, height=300, font=("Courier", 14))
            text_box.insert("0.0", text)
            text_box.configure(state="disabled")
            text_box.pack(expand=True, pady=10)
            ctk.CTkButton(
                frame, text="Apply Iterative Imputation", font=("Arial", 14),
                fg_color="#4a90e2", hover_color="#357abd", text_color="white",
                command=lambda: self.apply_imputation(imputer, numeric_cols)
            ).pack(pady=10)
        except Exception as e:
            ctk.CTkLabel(frame, text=f"Error: {str(e)}", text_color="red", font=("Arial", 14)).pack(expand=True)

    def apply_imputation(self, imputer, columns=None):
        try:
            if columns is None:
                self.processed_df = pd.DataFrame(imputer.fit_transform(self.processed_df), columns=self.processed_df.columns)
            else:
                self.processed_df[columns] = imputer.fit_transform(self.processed_df[columns])
            messagebox.showinfo("Success", "Imputation applied successfully.")
            self.populate_tabs()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply imputation: {str(e)}")

    def show_skewness(self, frame):
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if numeric_cols.empty:
                raise ValueError("No numeric columns available for skewness analysis.")
            skewness = self.df[numeric_cols].skew()
            text = f"Skewness of Numeric Columns:\n\n{skewness}"
            text_box = ctk.CTkTextbox(frame, width=600, height=300, font=("Courier", 14))
            text_box.insert("0.0", text)
            text_box.configure(state="disabled")
            text_box.pack(expand=True, pady=10)
        except Exception as e:
            ctk.CTkLabel(frame, text=f"Error: {str(e)}", text_color="red", font=("Arial", 14)).pack(expand=True)

    def show_log_transformation(self, frame):
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if numeric_cols.empty:
                raise ValueError("No numeric columns available for log transformation.")
            temp_df = self.df.copy()
            for col in numeric_cols:
                if (temp_df[col] > 0).all():
                    temp_df[col] = np.log1p(temp_df[col])
            text = "Log Transformation applied to positive numeric columns.\n\nPreview:\n" + str(temp_df[numeric_cols].head())
            text_box = ctk.CTkTextbox(frame, width=600, height=300, font=("Courier", 14))
            text_box.insert("0.0", text)
            text_box.configure(state="disabled")
            text_box.pack(expand=True, pady=10)
            ctk.CTkButton(
                frame, text="Apply Log Transformation", font=("Arial", 14),
                fg_color="#4a90e2", hover_color="#357abd", text_color="white",
                command=self.apply_log_transformation
            ).pack(pady=10)
        except Exception as e:
            ctk.CTkLabel(frame, text=f"Error: {str(e)}", text_color="red", font=("Arial", 14)).pack(expand=True)

    def apply_log_transformation(self):
        try:
            numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if (self.processed_df[col] > 0).all():
                    self.processed_df[col] = np.log1p(self.processed_df[col])
            messagebox.showinfo("Success", "Log transformation applied successfully.")
            self.populate_tabs()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply log transformation: {str(e)}")

    def evaluate_log_effect(self, frame):
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if numeric_cols.empty:
                raise ValueError("No numeric columns available.")
            temp_df = self.df.copy()
            skewness_before = temp_df[numeric_cols].skew()
            for col in numeric_cols:
                if (temp_df[col] > 0).all():
                    temp_df[col] = np.log1p(temp_df[col])
            skewness_after = temp_df[numeric_cols].skew()
            text = f"Skewness Before Log:\n{skewness_before}\n\nSkewness After Log:\n{skewness_after}"
            text_box = ctk.CTkTextbox(frame, width=600, height=300, font=("Courier", 14))
            text_box.insert("0.0", text)
            text_box.configure(state="disabled")
            text_box.pack(expand=True, pady=10)
        except Exception as e:
            ctk.CTkLabel(frame, text=f"Error: {str(e)}", text_color="red", font=("Arial", 14)).pack(expand=True)

    def show_boxplot(self, frame):
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if numeric_cols.empty:
                raise ValueError("No numeric columns available for boxplot.")
            fig, ax = plt.subplots(figsize=(8, 5))
            self.df[numeric_cols].boxplot(ax=ax)
            ax.set_title("Boxplot of Numeric Columns")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            plt.close(fig)
        except Exception as e:
            ctk.CTkLabel(frame, text=f"Error: {str(e)}", text_color="red", font=("Arial", 14)).pack(expand=True)

    def show_label_encoding(self, frame):
        try:
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
            if categorical_cols.empty:
                raise ValueError("No categorical columns available for label encoding.")
            temp_df = self.df.copy()
            le = LabelEncoder()
            for col in categorical_cols:
                temp_df[col] = le.fit_transform(temp_df[col].astype(str))
            text = "Label Encoding applied to categorical columns.\n\nPreview:\n" + str(temp_df[categorical_cols].head())
            text_box = ctk.CTkTextbox(frame, width=600, height=300, font=("Courier", 14))
            text_box.insert("0.0", text)
            text_box.configure(state="disabled")
            text_box.pack(expand=True, pady=10)
            ctk.CTkButton(
                frame, text="Apply Label Encoding", font=("Arial", 14),
                fg_color="#4a90e2", hover_color="#357abd", text_color="white",
                command=self.apply_label_encoding
            ).pack(pady=10)
        except Exception as e:
            ctk.CTkLabel(frame, text=f"Error: {str(e)}", text_color="red", font=("Arial", 14)).pack(expand=True)

    def apply_label_encoding(self):
        try:
            categorical_cols = self.processed_df.select_dtypes(include=['object', 'category']).columns
            le = LabelEncoder()
            for col in categorical_cols:
                self.processed_df[col] = le.fit_transform(self.processed_df[col].astype(str))
            messagebox.showinfo("Success", "Label encoding applied successfully.")
            self.populate_tabs()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply label encoding: {str(e)}")

    def draw_histogram(self, frame):
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if numeric_cols.empty:
                raise ValueError("No numeric columns available for histogram.")
            fig, ax = plt.subplots(figsize=(8, 5))
            self.df[numeric_cols[1]].hist(ax=ax, bins=20, color="#4a90e2")
            ax.set_title(f"Histogram of {numeric_cols[1]}")
            ax.set_xlabel(numeric_cols[1])
            ax.set_ylabel("Frequency")
            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            plt.close(fig)
        except Exception as e:
            ctk.CTkLabel(frame, text=f"Error: {str(e)}", text_color="red", font=("Arial", 14)).pack(expand=True)

    def show_standard_deviation(self, frame):
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if numeric_cols.empty:
                raise ValueError("No numeric columns available for standard deviation.")
            std = self.df[numeric_cols].std()
            text = f"Standard Deviation of Numeric Columns:\n\n{std}"
            text_box = ctk.CTkTextbox(frame, width=600, height=300, font=("Courier", 14))
            text_box.insert("0.0", text)
            text_box.configure(state="disabled")
            text_box.pack(expand=True, pady=10)
            ctk.CTkButton(
                frame, text="Apply Standard Scaling", font=("Arial", 14),
                fg_color="#4a90e2", hover_color="#357abd", text_color="white",
                command=self.apply_standard_scaling
            ).pack(pady=10)
        except Exception as e:
            ctk.CTkLabel(frame, text=f"Error: {str(e)}", text_color="red", font=("Arial", 14)).pack(expand=True)

    def apply_standard_scaling(self):
        try:
            numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
            if numeric_cols.empty:
                raise ValueError("No numeric columns available for scaling.")
            scaler = StandardScaler()
            self.processed_df[numeric_cols] = scaler.fit_transform(self.processed_df[numeric_cols])
            messagebox.showinfo("Success", "Standard scaling applied successfully.")
            self.populate_tabs()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply standard scaling: {str(e)}")

    def draw_histogram_after_std(self, frame):
        try:
            numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
            if numeric_cols.empty:
                raise ValueError("No numeric columns available for histogram.")
            temp_df = self.processed_df.copy()
            scaler = StandardScaler()
            temp_df[numeric_cols] = scaler.fit_transform(temp_df[numeric_cols])
            fig, ax = plt.subplots(figsize=(8, 5))
            temp_df[numeric_cols[1]].hist(ax=ax, bins=20, color="#4a90e2")
            ax.set_title(f"Histogram of {numeric_cols[1]} After Standard Scaling")
            ax.set_xlabel(numeric_cols[1])
            ax.set_ylabel("Frequency")
            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            plt.close(fig)
        except Exception as e:
            ctk.CTkLabel(frame, text=f"Error: {str(e)}", text_color="red", font=("Arial", 14)).pack(expand=True)

    def show_z_score(self, frame):
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if numeric_cols.empty:
                raise ValueError("No numeric columns available for Z-score.")
            temp_df = self.df.copy()
            temp_df[numeric_cols] = (temp_df[numeric_cols] - temp_df[numeric_cols].mean()) / temp_df[numeric_cols].std()
            text = "Z-Score Normalization applied.\n\nPreview:\n" + str(temp_df[numeric_cols].head())
            text_box = ctk.CTkTextbox(frame, width=600, height=300, font=("Courier", 14))
            text_box.insert("0.0", text)
            text_box.configure(state="disabled")
            text_box.pack(expand=True, pady=10)
            ctk.CTkButton(
                frame, text="Apply Z-Score Normalization", font=("Arial", 14),
                fg_color="#4a90e2", hover_color="#357abd", text_color="white",
                command=self.apply_z_score
            ).pack(pady=10)
        except Exception as e:
            ctk.CTkLabel(frame, text=f"Error: {str(e)}", text_color="red", font=("Arial", 14)).pack(expand=True)

    def apply_z_score(self):
        try:
            numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
            if numeric_cols.empty:
                raise ValueError("No numeric columns available for Z-score.")
            self.processed_df[numeric_cols] = (self.processed_df[numeric_cols] - self.processed_df[numeric_cols].mean()) / self.processed_df[numeric_cols].std()
            messagebox.showinfo("Success", "Z-score normalization applied successfully.")
            self.populate_tabs()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply Z-score: {str(e)}")

    def draw_histogram_after_zscore(self, frame):
        try:
            numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
            if numeric_cols.empty:
                raise ValueError("No numeric columns available for histogram.")
            temp_df = self.processed_df.copy()
            temp_df[numeric_cols] = (temp_df[numeric_cols] - temp_df[numeric_cols].mean()) / temp_df[numeric_cols].std()
            fig, ax = plt.subplots(figsize=(8, 5))
            temp_df[numeric_cols[1]].hist(ax=ax, bins=20, color="#4a90e2")
            ax.set_title(f"Histogram of {numeric_cols[1]} After Z-Score Normalization")
            ax.set_xlabel(numeric_cols[1])
            ax.set_ylabel("Frequency")
            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            plt.close(fig)
        except Exception as e:
            ctk.CTkLabel(frame, text=f"Error: {str(e)}", text_color="red", font=("Arial", 14)).pack(expand=True)

    def draw_histogram_after_median(self, frame):
        try:
            numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
            if numeric_cols.empty:
                raise ValueError("No numeric columns available for histogram.")
            temp_df = self.processed_df.copy()
            imputer = SimpleImputer(strategy="median")
            temp_df[numeric_cols] = imputer.fit_transform(temp_df[numeric_cols])
            fig, ax = plt.subplots(figsize=(8, 5))
            temp_df[numeric_cols[1]].hist(ax=ax, bins=20, color="#4a90e2")
            ax.set_title(f"Histogram of {numeric_cols[1]} After Median Imputation")
            ax.set_xlabel(numeric_cols[1])
            ax.set_ylabel("Frequency")
            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            plt.close(fig)
        except Exception as e:
            ctk.CTkLabel(frame, text=f"Error: {str(e)}", text_color="red", font=("Arial", 14)).pack(expand=True)