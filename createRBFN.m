%% CREATERBFN.M
% Constructs a RBFN to be used for the evaluation over the test set.

function [Y_predicted, fScore, accuracy, ...
          C_best, s_best, w_best] = createRBFN(X, Y, X_test, Y_test, ...
                                         vecM, regularised)
	
	% Obtain the input matrix dimensions.
	N = size(X, 1);
	N_test = size(X_test, 1);
	numClasses = size(Y, 2);

	lenM = length(vecM);
	fScore = zeros(lenM, 1);
	accuracy = zeros(lenM, 1);

	accuracy_best = 0;
	C_best = 0;
	s_best = 0;
	w_best = cell(1, numClasses);

	Y_predicted = cell(1, lenM);

	for iter = 1 : lenM
		M = vecM(iter);
		Y_predicted{iter} = zeros(N_test, numClasses);

		fprintf('Performing RBFN classification with %d centres ...\n', M);

		% Randomly select M training data points as centres.
		cInds = randperm(N);
		cInds = cInds(1 : M);

		C = X(cInds, :);

		% Calculate the pair-wise distance matrix D of the centres stored in C.
		D = pdist(C, 'euclidean');
		r_max = max(D(:));

		% Clean-up the distance matrix since it is not needed anymore.
		clear D;

		% Heuristically compute the corresponding spread parameter value.
		s = -(M / (r_max ^ 2));

		% Construct the Phi matrix.
		Phi = zeros(N, M);

		NM = N * M;
		NM_inds = 1 : NM;

		iNM_inds = mod(NM_inds, N);
		iNM_inds(iNM_inds == 0) = N;
		jNM_inds = ceil(NM_inds / N);

		fprintf('(M = %d) Creating the RBFN and assessing its performance on the corresponding test set ...\n', M);

		parfor k = 1 : NM
			i = iNM_inds(k);
			j = jNM_inds(k);

			Phi(k) = gaussianRBF(X(i, :), C(j, :), s);
		end

		%% Optimal (non-regularised) least-squares solution.
		if (~regularised)
			w_LS = cell(1, numClasses);

			pinvPhi = pinv(Phi);

			for k = 1 : numClasses
			w_LS{k} =  pinvPhi * Y(:, k);
			end

			Phi_test = zeros(N_test, M);

			NtM = N_test * M;
			NtM_inds = 1 : NtM;

			iInds = mod(NtM_inds, N_test);
			iInds(iInds == 0) = N_test;
			jInds = ceil(NtM_inds / N_test);

			parfor k = 1 : NtM
				i = iInds(k);
				j = jInds(k);
				Phi_test(k) = gaussianRBF(X_test(i, :), C(j, :), s);
			end

			y_predicted = cell(1, numClasses);

			for k = 1 : numClasses
				y_predicted{k} = Phi_test * w_LS{k};
			end

			Y_predicted{iter}(:, 1) = y_predicted{1} > y_predicted{2};
			Y_predicted{iter}(:, 2) = ~Y_predicted{iter}(:, 1);

			% Columns are the 6 different metrics (F-score, Phi-score, accuracy,
			% false positive rate, sensitivity and positive predictive value)
			results = evaluate(Y_predicted{iter}(:, 1), Y_test(:, 1));

			fScore(iter) = results{2}(1);
			accuracy(iter) = results{2}(3);

			fprintf(['[%d, %f] *** Optimal non-regularised least squares solution ***\n', ...
			'\tF-score: %f\n', ...
			'\tAccuracy: %f\n'], ...
			M, s, ... 
			fScore(iter), accuracy(iter));

		%% Optimal regularised least-squares solution.
		else
			G = zeros(M, M);

			MM = M * M;
			MM_inds = 1 : MM;

			iMM_inds = mod(MM_inds, M);
			iMM_inds(iMM_inds == 0) = M;
			jMM_inds = ceil(MM_inds / M);

			parfor k = 1 : MM
				i = iMM_inds(k);
				j = jMM_inds(k);
				G(k) = gaussianRBF(C(i, :), C(j, :), s);
			end

			lambda = 0.5;
			w_regLS = cell(1, numClasses);

			for k = 1 : numClasses
				w_regLS{k} = pinv(Phi' * Phi + lambda * G) * (Phi' * Y(:, k));
			end

			Phi_reg_test = zeros(N_test, M);

			NtM = N_test * M;
			NtM_inds = 1 : NtM;

			iNtM_inds = mod(NtM_inds, N_test);
			iNtM_inds(iNtM_inds == 0) = N_test;
			jNtM_inds = ceil(NtM_inds / N_test);

			parfor k = 1 : NtM
				i = iNtM_inds(k);
				j = jNtM_inds(k);
				Phi_reg_test(k) = gaussianRBF(X_test(i, :), C(j, :), s);
			end

			y_reg_predicted = cell(1, numClasses);

			for k = 1 : numClasses
				y_reg_predicted{k} = Phi_reg_test * w_regLS{k};
			end

			Y_predicted{iter}(:, 1) = y_reg_predicted{1} > y_reg_predicted{2};
			Y_predicted{iter}(:, 2) = ~Y_predicted{iter}(:, 1);

			% Columns are the 6 different metrics (F-score, Phi-score, accuracy,
			% false positive rate, sensitivity and positive predictive value)
			results_reg = evaluate(Y_predicted{iter}(:, 1), Y_test(:, 1));

			fScore(iter) = results_reg{2}(1);
			accuracy(iter) = results_reg{2}(3);

			if (accuracy(iter) > accuracy_best)
				C_best = C;
				s_best = s;
				w_best = w_regLS;
			end

			fprintf(['[%d, %f] *** Optimal regularised least squares solution (lambda = %f) ***\n', ...
			'\tF-score: %f\n', ...
			'\tAccuracy: %f\n'], ...
			M, s, lambda, ...
			fScore(iter), accuracy(iter));
		end
	end
end
