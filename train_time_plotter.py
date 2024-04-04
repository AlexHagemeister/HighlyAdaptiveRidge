import altair as alt
import pandas as pd

class TrainTimePlotter:
    """
    Class to plot training time vs. sample size

    PARAMS:
    data: pd.DataFrame w/ cols ['Sample Size', 'Method', 'training_time', 'MSE'] ("long" format) from RunTrials.run_trials
    d: int, dimension of the covariates (optional)
    dgp_type: str, type of DGP used to generate data (optional)
    *d and dgp_type are used to add a subtitle to the plots
    """

    @classmethod
    def plot(cls, data, d=None, dgp_type=None):

        plot_title = 'Training Time vs. Sample Size'
        if d and dgp_type:
            plot_title += f"\n(d={d}, {dgp_type} dgp)"


        # Aggregate results by sample size and method
        aggregated_df = data.groupby(['Sample Size', 'Method']).agg(
            mean_training_time=pd.NamedAgg(column='training_time', aggfunc='mean'),
            std_training_time=pd.NamedAgg(column='training_time', aggfunc='std')
        ).reset_index()

        # Create the line plot for the filtered data
        chart = alt.Chart(aggregated_df).mark_line(point=True).encode(
            x=alt.X('Sample Size:Q', title='Sample Size'),
            y=alt.Y('mean_training_time:Q', title='Mean Training Time (s)'),
            color=alt.Color('Method:N', legend=alt.Legend(title='Method')),
            tooltip=['Sample Size', 'Method', 'mean_training_time']
        ).properties(
            width=600,
            height=400
        ).configure_axis(
            grid=True
        ).configure_legend(
            titleFontSize=12,
            labelFontSize=10
        ).configure_title(
            fontSize=16
        ).properties(
            title=plot_title
        )
        return chart
