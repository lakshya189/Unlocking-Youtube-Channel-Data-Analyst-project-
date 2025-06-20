import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]

def load_and_prepare_data():
    """Load and preprocess the YouTube analytics data."""
    print("Loading and preparing data...")
    try:
        # Load the dataset
        df = pd.read_csv('youtube_channel_real_performance_analytics.csv')
        
        # Convert date columns
        df['Video Publish Time'] = pd.to_datetime(df['Video Publish Time'])
        
        # Extract date features
        df['Year'] = df['Video Publish Time'].dt.year
        df['Month'] = df['Video Publish Time'].dt.month
        df['Day'] = df['Video Publish Time'].dt.day
        df['DayOfWeek'] = df['Video Publish Time'].dt.dayofweek
        
        # Calculate engagement rate
        engagement_metrics = ['Likes', 'Comments', 'Shares', 'Subscribers']
        for metric in engagement_metrics:
            if metric in df.columns:
                df[f'{metric}_per_view'] = df[metric] / df['Views']
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_visualizations(df):
    """Create visualizations for YouTube channel analysis."""
    print("\nCreating visualizations...")
    try:
        # Create visualizations directory
        import os
        os.makedirs('visualizations', exist_ok=True)
        
        # 1. Monthly Views Over Time
        plt.figure(figsize=(14, 6))
        monthly_views = df.set_index('Video Publish Time')['Views'].resample('M').sum()
        monthly_views.plot()
        plt.title('Monthly Views Over Time')
        plt.ylabel('Total Views')
        plt.tight_layout()
        plt.savefig('visualizations/monthly_views.png')
        plt.close()
        
        # 2. Top 10 Videos by Views
        top_videos = df.nlargest(10, 'Views').copy()
        top_videos['ID'] = top_videos['ID'].astype(str)
        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_videos, x='Views', y='ID', palette='viridis')
        plt.title('Top 10 Videos by Views')
        plt.tight_layout()
        plt.savefig('visualizations/top_videos.png')
        plt.close()
        
        # 3. Correlation Heatmap
        corr_columns = ['Views', 'Watch Time (hours)', 'Subscribers', 
                       'Revenue per 1000 Views (USD)', 'Video Thumbnail CTR (%)']
        corr_columns = [col for col in corr_columns if col in df.columns]
        
        if len(corr_columns) >= 2:
            corr = df[corr_columns].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt=".2f")
            plt.title('Correlation Heatmap of Key Metrics')
            plt.tight_layout()
            plt.savefig('visualizations/correlation_heatmap.png')
            plt.close()
        
        # 4. Video Duration vs. Views
        if 'Video Duration' in df.columns:
            plt.figure(figsize=(12, 7))
            sns.scatterplot(
                data=df, 
                x='Video Duration', 
                y='Views', 
                alpha=0.6, 
                hue='Video Thumbnail CTR (%)',
                palette='viridis'
            )
            plt.title('Video Duration vs. Views (Color by CTR %)')
            plt.tight_layout()
            plt.savefig('visualizations/duration_vs_views.png')
            plt.close()
        
        # 5. Revenue Analysis
        if 'Estimated Revenue (USD)' in df.columns:
            plt.figure(figsize=(12, 7))
            sns.scatterplot(
                data=df, 
                x='Views', 
                y='Estimated Revenue (USD)', 
                alpha=0.6, 
                hue='Revenue per 1000 Views (USD)',
                palette='plasma'
            )
            plt.title('Views vs. Estimated Revenue')
            plt.tight_layout()
            plt.savefig('visualizations/revenue_analysis.png')
            plt.close()
            
        print("Visualizations created successfully!")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def train_prediction_model(df):
    """Train a prediction model to estimate video views."""
    print("\nTraining prediction model...")
    try:
        # Select features and target
        features = [
            'Video Duration', 'DayOfWeek', 'Month', 'Year',
            'Watch Time (hours)', 'Subscribers', 'Video Thumbnail CTR (%)'
        ]
        
        # Only keep features that exist in the dataframe
        features = [f for f in features if f in df.columns]
        
        if len(features) < 2:
            print("Not enough features for modeling")
            return None, None, None
            
        X = df[features].copy()
        y = df['Views']
        
        # Handle missing values
        X = X.fillna(X.median())
        y = y.fillna(y.median())
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model trained successfully!")
        print(f"Mean Squared Error: {mse:,.2f}")
        print(f"R² Score: {r2:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('visualizations/feature_importance.png')
        plt.close()
        
        return model, scaler, feature_importance
        
    except Exception as e:
        print(f"Error training model: {e}")
        return None, None, None

def generate_report(df, model, feature_importance):
    """Generate a comprehensive report of the analysis."""
    print("\nGenerating report...")
    try:
        report = []
        
        # Basic statistics
        report.append("=== YouTube Channel Analysis Report ===\n")
        report.append(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Channel overview
        report.append("=== Channel Overview ===")
        report.append(f"Total Videos: {len(df)}")
        report.append(f"Total Views: {df['Views'].sum():,}")
        report.append(f"Total Subscribers: {df['Subscribers'].sum():,}")
        report.append(f"Average Video Duration: {df['Video Duration'].mean()/60:.1f} minutes\n")
        
        # Top performing videos
        report.append("=== Top Performing Videos ===")
        top_videos = df.nlargest(5, 'Views')[['ID', 'Views', 'Watch Time (hours)', 'Subscribers']]
        report.append(top_videos.to_string())
        
        # Key metrics
        report.append("\n=== Key Metrics ===")
        metrics = {
            'Average Views per Video': df['Views'].mean(),
            'Median Views per Video': df['Views'].median(),
            'Average Watch Time (hours)': df['Watch Time (hours)'].mean(),
            'Average CTR (%)': df['Video Thumbnail CTR (%)'].mean(),
            'Average Revenue per 1000 Views (USD)': df['Revenue per 1000 Views (USD)'].mean()
        }
        
        for metric, value in metrics.items():
            if 'USD' in metric:
                report.append(f"{metric}: ${value:,.2f}")
            elif 'Time' in metric:
                report.append(f"{metric}: {value:,.1f}")
            else:
                report.append(f"{metric}: {value:,.0f}")
        
        # Model insights
        if model is not None and feature_importance is not None:
            report.append("\n=== Model Insights ===")
            report.append("Feature Importance (higher is better):")
            report.append(feature_importance[['Feature', 'Importance']].to_string(index=False))
            report.append("\nRecommendations based on analysis:")
            report.append("1. Focus on creating content similar to your top-performing videos")
            report.append("2. Optimize video duration based on the duration vs. views analysis")
            report.append("3. Improve thumbnails and titles to increase CTR")
            report.append("4. Consider the best days/times to publish based on historical performance")
        
        # Save report
        with open('youtube_channel_analysis_report.txt', 'w') as f:
            f.write('\n'.join(report))
            
        print("Report generated successfully!")
        
    except Exception as e:
        print(f"Error generating report: {e}")

def main():
    """Main function to run the analysis."""
    # Load and prepare data
    df = load_and_prepare_data()
    if df is None:
        return
    
    # Create visualizations
    create_visualizations(df)
    
    # Train prediction model
    model, scaler, feature_importance = train_prediction_model(df)
    
    # Generate report
    generate_report(df, model, feature_importance)
    
    print("\n=== Analysis Complete! ===")
    print("Check the 'visualizations' folder for charts and 'youtube_channel_analysis_report.txt' for the full report.")

if __name__ == "__main__":
    main()
