from ApproxMethods.BasicMethods.PermutationSampling import PermutationSampling
from ApproxMethods.KernelSHAP.KernelShap import KernelShap
from ApproxMethods.KernelSHAP.UnbiasedSHAPCovert import UnbiasedKernelSHAP
from ApproxMethods.OwenSampling.OwenSampling import OwenSampling
from ApproxMethods.SVARM.StratifiedSVARM import StratifiedSVARM
from ApproxMethods.SVARM.SVARM import SVARM
from ApproxMethods.SVARM.StratifiedSVARMnoReplacement import StratifiedSVARMnoReplacement
from ApproxMethods.StratifiedSampling.StratifiedSampling import StratifiedSampling
from ApproxMethods.StructuredSampling.StructuredSampling import StructuredSampling

from Games.LookUpGame import LookUpGame

from run_svarm_experiment import run_experiment


if __name__ == "__main__":
    approx_methods = [
        PermutationSampling(),
        StructuredSampling(),

        # paper version: set normalize=False, warm_up=True
        SVARM(normalize=False, warm_up=True),

        # paper version: set normalize=False, warm_up=True, rich_warm_up=False, paired_sampling=False
        # dist_type sets the sampling distribution over coalition sizes: paper for the theoretically derived one, uniform for the uniform distribution
        StratifiedSVARM(normalize=False, warm_up=True, rich_warm_up=False, paired_sampling=False, dist_type="paper"),
        StratifiedSVARM(normalize=False, warm_up=True, rich_warm_up=False, paired_sampling=False, dist_type="uniform"),

        # paper version: normalize=False, dynamic=True
        # dist_type sets the sampling distribution over coalition sizes: paper for the theoretically derived one, uniform for the uniform distribution
        # smart_factor speeds up the sampling process without replacement by changing from a list of sampled coalitions to a list of coalitions to sample, values between 0.2 and 0.8 are reasonable, has no effect on the app
        StratifiedSVARMnoReplacement(normalize=False, dynamic=True, dist_type="paper", smart_factor=0.5),
        StratifiedSVARMnoReplacement(normalize=False, dynamic=True, dist_type="uniform", smart_factor=0.5),
    ]

    approx_methods_extra = [
        # paper version: set normalize=False
        KernelShap(normalize=False),
        UnbiasedKernelSHAP(normalize=False),

        StratifiedSampling(),
        OwenSampling()
    ]

    # -------------------- CONSTANTS --------------------- #
    BUDGET = 2**14
    NUMBER_OF_RUNS = 100
    STEP_SIZE = 100
    STEP_SIZE_RERUN_APPROXIMATOR = 100
    STEPS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    STEPS += list(range(2000, BUDGET + 1000, 1000))

    # NLP game -------------------------------------------------------------------------------------

    game_list = []
    used_ids = set()
    data_folder = "image_classifier"
    for i in range(NUMBER_OF_RUNS):
        game = LookUpGame(data_folder=data_folder, n=14, set_zero=True,
                          used_ids=used_ids)
        game_list.append(game)
        used_ids = game.used_ids

    run_experiment(
        game_list=game_list,
        approx_methods=approx_methods,
        approx_methods_extra=approx_methods_extra,
        budget=BUDGET,
        number_of_runs=NUMBER_OF_RUNS,
        step_size=STEP_SIZE,
        step_size_rerun_approximator=STEP_SIZE_RERUN_APPROXIMATOR,
        use_exact_same_game=False,
        steps=STEPS
    )
