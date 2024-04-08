import benchmark.mix.mix_params_exp as mix_params_exp
import benchmark.mix.mix_baseline_exp as mix_base_exp
import benchmark.mix.mix_final_exp as mix_final_exp
import argparse


data_choices = mix_base_exp.all_data + ['all']
params_choices = mix_params_exp.all_params + ['all']
baselines_choices = mix_base_exp.all_baselines + ['all']
version_choices = mix_final_exp.all_versions + ['all']

parser = argparse.ArgumentParser(description='mix-exp-v1.0')
parser.add_argument('-t', '--type', help='Experiment type', choices=['params', 'baselines', 'final'])
parser.add_argument('-d', '--data', nargs='+', help='Data for evaluation', choices=data_choices)
parser.add_argument('-rl', '--run_label', help='Run label')

parser.add_argument('-dev', '--device', help='Device')
parser.add_argument('-p', '--params', nargs='*', help='Parameters to evaluate', choices=params_choices)
parser.add_argument('-b', '--baselines', nargs='*', help='Baselines to evaluate', choices=baselines_choices)
parser.add_argument('-lr', '--baselines_lrs', nargs='*', type=float, help='Baselines learning rates')
parser.add_argument('-bn', '--baselines_bns', nargs='*', type=int, help='Baselines batch normalization on/off')
parser.add_argument('-v', '--versions', nargs='*', help='Versions of MIX to evaluate', choices=version_choices)


def run():
    args = parser.parse_args()

    if args.type == 'params':
        print(f'Running parameters evaluation: {args.params} {args.data} {args.run_label} {args.device}')
        mix_params_exp.run(args.params, args.data, args.run_label, args.device)
    elif args.type == 'baselines':
        print(f'Running baselines evaluation: {args.baselines} {args.baselines_lrs} {args.baselines_bns} {args.data} '
              f'{args.run_label} {args.device}')
        mix_base_exp.run(args.baselines, args.baselines_lrs, args.baselines_bns, args.data, args.run_label,
                          args.device)
    elif args.type == 'final':
        print(f'Running final MIX evaluation: {args.versions} {args.data} {args.run_label} {args.device}')
        mix_final_exp.run(args.versions, args.data, args.run_label, args.device)


if __name__ == '__main__':
    run()
