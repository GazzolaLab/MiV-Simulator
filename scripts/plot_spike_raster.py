
import os, sys, gc
import click
import biophys_microcircuit
from biophys_microcircuit import plot, utils

script_name = os.path.basename(__file__)

@click.command()
@click.option("--spike-events-path", '-p', required=True, type=click.Path())
@click.option("--spike-events-namespace", '-n', type=str, default='Spike Data')
@click.option("--populations", '-i', type=str, multiple=True)
@click.option("--max-spikes", type=int, default=int(1e6))
@click.option("--spike-hist-bin", type=float, default=5.0)
@click.option("--t-variable", type=str, default='t')
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--font-size", type=float, default=14)
@click.option("--fig-size", type=(float,float), default=(15,8))
@click.option("--labels", type=str, default='legend')
@click.option("--save-format", type=str, default='png')
@click.option("--include-artificial/--exclude-artificial", type=bool, default=True, is_flag=True)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(spike_events_path, spike_events_namespace, populations, max_spikes, spike_hist_bin, t_variable, t_max, t_min, font_size, fig_size, labels, save_format, include_artificial, verbose):

    utils.config_logging(verbose)
    
    if t_max is None:
        time_range = None
    else:
        if t_min is None:
            time_range = [0.0, t_max]
        else:
            time_range = [t_min, t_max]

    if not populations:
        populations = ['eachPop']
        
    plot.plot_spike_raster (spike_events_path, spike_events_namespace, include=populations, time_range=time_range, time_variable=t_variable, pop_rates=True, spike_hist='subplot', max_spikes=max_spikes, spike_hist_bin=spike_hist_bin, include_artificial=include_artificial, fontSize=font_size, figSize=fig_size, labels=labels, saveFig=True, figFormat=save_format)
    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):])
