
dataset_name = None

import optuna
from optuna.trial import TrialState
from attacks.opt_objectives import square_objective, evo_objective, zoo_objective


if __name__ == '__main__':

    study = optuna.create_study(direction="minimize")
    study.optimize(zoo_objective, n_trials=50, timeout=None)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
