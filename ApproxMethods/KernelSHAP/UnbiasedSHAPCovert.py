import numpy as np

from ApproxMethods.AbstractApproxMethod import BaseApproxMethod


class GameWrapper:

    def __init__(self, game):
        self.game = game
        self.players = game.n

    def __call__(self, S):
        result = np.empty(S.shape[0])
        for i in range(S.shape[0]):
            sample = S[i]
            s_set = set()
            for j, value in enumerate(sample):
                if value == 1:
                    s_set.add(j)
            result[i] = self.game.get_value(s_set)
        return result

    def grand(self):
        '''Get grand coalition value.'''
        return self.__call__(np.ones((1, self.players), dtype=bool))[0]

    def null(self):
        '''Get null coalition value.'''
        return self.__call__(np.zeros((1, self.players), dtype=bool))[0]

# Unbiased KernelSHAP (Covert and Lee., 2021)
class UnbiasedKernelSHAP(BaseApproxMethod):

    def __init__(self, normalize, pairing=False):
        super().__init__(normalize)
        self.pairing = pairing
        self.sv_values = np.zeros(self.n)

    def get_estimates(self):
        return self.sv_values

    def get_name(self) -> str:
        if self.pairing:
            return "UnbiasedKernelSHAP_Covert_Pairing"
        return "UnbiasedKernelSHAP_Covert"

    def approximate_shapley_values(self) -> dict:

        u_ksh_covert_sv, _, _ = self.calculate_unbiased_kernel_shap(
            game=GameWrapper(self.game),
            batch_size=self.budget,
            detect_convergence=False,
            n_samples=self.budget,
            paired_sampling=self.pairing
        )
        self.sv_values = u_ksh_covert_sv

        self.experiment_storage.add_shapley_values(self.get_estimates())
        return self.experiment_storage.to_json()

    def calculate_unbiased_kernel_shap(
            self,
            game,
            batch_size=512,
            detect_convergence=False,
            thresh=0.01,
            n_samples=None,
            paired_sampling=True,
            return_all=False,
            verbose=False
    ):

        # Possibly force convergence detection.
        if n_samples is None:
            n_samples = 1e20
            if not detect_convergence:
                detect_convergence = True
                if verbose:
                    print('Turning convergence detection on')

        if detect_convergence:
            assert 0 < thresh < 1

        # Weighting kernel (probability of each subset size).
        num_players = game.players
        weights = np.arange(1, num_players)
        weights = 1 / (weights * (num_players - weights))
        weights = weights / np.sum(weights)

        null = game.null()
        grand = game.grand()

        # Calculate difference between grand and null coalitions.
        total = grand - null

        # Set up bar.
        n_loops = int(np.ceil(n_samples / batch_size))

        # Setup.
        A = self._calculate_A(num_players)
        n = 0
        b = 0
        b_sum_squares = 0

        # For tracking progress.
        if return_all:
            N_list = []
            std_list = []
            val_list = []

        all_S = []
        # Begin sampling.
        for it in range(n_loops):
            # Sample subsets.
            S = np.zeros((batch_size, num_players), dtype=bool)
            num_included = np.random.choice(num_players - 1, size=batch_size,
                                            p=weights) + 1
            for row, num in zip(S, num_included):
                inds = np.random.choice(num_players, size=num, replace=False)
                row[inds] = 1
            all_S.append(S)

            S_set = self._turn_bool_s_into_set(S[0])
            # Update estimators.
            if paired_sampling:
                # Paired samples.
                game_eval = game(S) - null
                S_comp = np.logical_not(S)
                S_comp_set = self._turn_bool_s_into_set(S_comp[0])
                comp_eval = game(S_comp) - null
                b_sample = 0.5 * (
                        S.astype(float).T * game_eval[:, np.newaxis].T
                        + S_comp.astype(float).T * comp_eval[:, np.newaxis].T).T
            else:
                # Single sample.
                b_sample = (S.astype(float).T
                            * (game(S) - null)[:, np.newaxis].T).T

            # Welford's algorithm.
            n += batch_size
            b_diff = b_sample - b
            b += np.sum(b_diff, axis=0) / n
            b_diff2 = b_sample - b
            b_sum_squares += np.sum(
                np.expand_dims(b_diff, 2) * np.expand_dims(b_diff2, 1),
                axis=0)

            # Calculate progress.
            values, std = self._calculate_exact_result(A, b, total, b_sum_squares, n)
            ratio = np.max(
                np.max(std, axis=0) / (values.max(axis=0) - values.min(axis=0)))
            self.sv_values = values

            # Print progress message.
            if verbose:
                if detect_convergence:
                    print(f'StdDev Ratio = {ratio:.4f} (Converge at {thresh:.4f})')
                else:
                    print(f'StdDev Ratio = {ratio:.4f}')

            # Check for convergence.
            if detect_convergence:
                if ratio < thresh:
                    if verbose:
                        print('Detected convergence')
                    break

            # Forecast number of iterations required.
            if detect_convergence:
                N_est = (it + 1) * (ratio / thresh) ** 2

            # Save intermediate quantities.
            if return_all:
                val_list.append(values)
                std_list.append(std)
                if detect_convergence:
                    N_list.append(N_est)

        # Return results.
        if return_all:
            # Dictionary for progress tracking.
            iters = (
                    (np.arange(it + 1) + 1) * batch_size *
                    (1 + int(paired_sampling)))
            tracking_dict = {
                'values': val_list,
                'std': std_list,
                'iters': iters}
            if detect_convergence:
                tracking_dict['N_est'] = N_list

            return values, std, all_S
        else:
            return values, std, all_S

    @staticmethod
    def _calculate_A(num_players):
        '''Calculate A parameter's exact form.'''
        p_coaccur = (
                (np.sum((np.arange(2, num_players) - 1)
                        / (num_players - np.arange(2, num_players)))) /
                (num_players * (num_players - 1) *
                 np.sum(1 / (np.arange(1, num_players)
                             * (num_players - np.arange(1, num_players))))))
        A = np.eye(num_players) * 0.5 + (1 - np.eye(num_players)) * p_coaccur
        return A

    @staticmethod
    def _calculate_exact_result(A, b, total, b_sum_squares, n):
        '''Calculate the regression coefficients and uncertainty estimates.'''
        num_players = A.shape[1]
        scalar_values = len(b.shape) == 1
        if scalar_values:
            A_inv_one = np.linalg.solve(A, np.ones(num_players))
        else:
            A_inv_one = np.linalg.solve(A, np.ones((num_players, 1)))
        A_inv_vec = np.linalg.solve(A, b)
        values = (
                A_inv_vec -
                A_inv_one * (np.sum(A_inv_vec, axis=0, keepdims=True) - total)
                / np.sum(A_inv_one))

        # Calculate variance.
        try:
            # Symmetrize, rescale and swap axes.
            b_sum_squares = 0.5 * (b_sum_squares
                                   + np.swapaxes(b_sum_squares, 0, 1))
            b_cov = b_sum_squares / (n ** 2)
            if scalar_values:
                b_cov = np.expand_dims(b_cov, 0)
            else:
                b_cov = np.swapaxes(b_cov, 0, 2)

            # Use Cholesky decomposition to calculate variance.
            cholesky = np.linalg.cholesky(b_cov)
            L = (
                    np.linalg.solve(A, cholesky) +
                    np.matmul(np.outer(A_inv_one, A_inv_one), cholesky)
                    / np.sum(A_inv_one))
            beta_cov = np.matmul(L, np.moveaxis(L, -2, -1))
            var = np.diagonal(beta_cov, offset=0, axis1=-2, axis2=-1)
            std = var ** 0.5
            if scalar_values:
                std = std[0]
            else:
                std = std.T
        except np.linalg.LinAlgError:
            # b_cov likely is not PSD due to insufficient samples.
            std = np.ones(num_players) * np.nan

        return values, std

    @staticmethod
    def _turn_bool_s_into_set(S):
        S_set = set()
        for j, value in enumerate(S):
            if value == 1:
                S_set.add(j)
        return S_set
