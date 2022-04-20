import os, sys, click
import dentate
from MiV import plot, utils
from MiV.utils import Context, is_interactive

script_name = os.path.basename(__file__)

context = Context()


@click.command()
@click.option("--config-file", required=False, type=str)
@click.option("--config-prefix", required=False, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='config')
@click.option("--input-path", '-p', required=True, type=click.Path())
@click.option("--spike-namespace", type=str, default='Spike Events')
@click.option("--state-namespace", type=str, default='Intracellular soma')
@click.option("--populations", '-i', type=str, multiple=True)
@click.option("--include-artificial/--exclude-artificial", default=True, is_flag=True)
@click.option("--target-input-features-path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--target-input-features-namespace", type=str, default='Selectivity Features')
@click.option("--target-input-features-arena-id", type=str)
@click.option("--target-input-features-trajectory-id", type=str)
@click.option("--gid", '-g', type=int)
@click.option("--n-trials", '-n', type=int, default=-1)
@click.option("--spike-hist-bin", type=float, default=5.0)
@click.option("--all-spike-hist", is_flag=True)
@click.option("--labels", type=str, default='overlay')
@click.option("--lowpass-plot-type", type=str, default='overlay')
@click.option("--legend", type=str, default='overlay')
@click.option("--state-variable", type=str, default='v')
@click.option("--t-variable", type=str, default='t')
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--font-size", type=float, default=14)
@click.option("--line-width", type=int, default=1)
@click.option("--verbose", "-v", is_flag=True)
def main(config_file, config_prefix, input_path, spike_namespace, state_namespace, populations, include_artificial,
         target_input_features_path, target_input_features_namespace,
         target_input_features_arena_id, target_input_features_trajectory_id,
         gid, n_trials, spike_hist_bin, all_spike_hist,
         labels, lowpass_plot_type, legend, state_variable, t_variable, t_max, t_min, font_size, line_width, verbose):

    utils.config_logging(verbose)
    
    if t_max is None:
        time_range = None
    else:
        if t_min is None:
            time_range = [0.0, t_max]
        else:
            time_range = [t_min, t_max]

    if len(populations) == 0:
        populations = ['eachPop']

    plot.plot_network_clamp(input_path, spike_namespace, state_namespace, gid=gid,
                            target_input_features_path=target_input_features_path,
                            target_input_features_namespace=target_input_features_namespace,
                            target_input_features_arena_id=target_input_features_arena_id,
                            target_input_features_trajectory_id=target_input_features_trajectory_id,
                            config_prefix=config_prefix, config_file=config_file,
                            include=populations, include_artificial=include_artificial,
                            time_range=time_range, time_variable=t_variable, intracellular_variable=state_variable,
                            all_spike_hist=all_spike_hist, spike_hist_bin=spike_hist_bin, labels=labels, 
                            lowpass_plot_type=lowpass_plot_type, n_trials=n_trials, fontSize=font_size, legend=legend, 
                            saveFig=True, lw=line_width)

    if is_interactive:
        context.update(locals())


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):],
         standalone_mode=False)
