# YouTube Channel Performance Analysis

This project provides comprehensive analytics and insights for YouTube channel performance. It analyzes various metrics including views, engagement, revenue, and more to help channel owners make data-driven decisions.

## Features

- Comprehensive YouTube analytics dashboard
- Predictive modeling for video performance
- Detailed visualization reports
- Automated performance analysis
- Feature importance analysis
- Actionable recommendations

## Requirements

See [requirements.txt](requirements.txt) for all dependencies.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your YouTube analytics data in `youtube_channel_real_performance_analytics.csv`

3. Run the analysis:
```bash
python complete_youtube_analysis.py
```

## Project Structure

- `complete_youtube_analysis.py`: Main analysis script
- `youtube_channel_real_performance_analytics.csv`: Input data file
- `visualizations/`: Directory for generated plots
- `youtube_channel_analysis_report.txt`: Generated analysis report

## Key Visualizations

### Monthly Views Trend
![Monthly Views Over Time](https://github.com/lakshya189/Unlocking-Youtube-Channel-Data-Analyst-project-/blob/master/visualizations/monthly_views.png)

### Top Performing Videos
![Top 10 Videos by Views](https://github.com/lakshya189/Unlocking-Youtube-Channel-Data-Analyst-project-/blob/master/visualizations/top_videos.png)

### Correlation Analysis
![Correlation Heatmap](https://github.com/lakshya189/Unlocking-Youtube-Channel-Data-Analyst-project-/blob/master/visualizations/correlation_heatmap.png)

### Video Duration vs Views
![Duration vs Views](https://github.com/lakshya189/Unlocking-Youtube-Channel-Data-Analyst-project-/blob/master/visualizations/duration_vs_views.png)

### Feature Importance
![Feature Importance](https://github.com/lakshya189/Unlocking-Youtube-Channel-Data-Analyst-project-/blob/master/visualizations/feature_importance.png)

### Revenue Analysis
![Revenue Analysis](https://github.com/lakshya189/Unlocking-Youtube-Channel-Data-Analyst-project-/blob/master/visualizations/revenue_analysis.png)

## Output

The script generates:
- Visualizations in the `visualizations/` directory
- A comprehensive analysis report in `youtube_channel_analysis_report.txt`
- Predictive model insights and feature importance analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.
