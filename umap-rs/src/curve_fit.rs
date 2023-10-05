use crate::{
    func_1d::Func1D,
    utils::{lu_decomp, lu_matrix_solve, matrix_solve},
};
use log::info;
use ndarray::{s, Array1, Array2};

/// Figure of merit that is minimized during the fit procedure
pub fn chi2(y: &Array1<f64>, y_model: &Array1<f64>, sy: &Array1<f64>) -> f64 {
    ((y - y_model) / sy).map(|x| x.powi(2)).sum()
}

/// Contains all relevant information after one minimization step
pub struct MinimizationStep {
    parameters: Array1<f64>,
    delta: Array1<f64>,
    y_model: Array1<f64>,
    chi2: f64,
    red_chi2: f64,
    metric: f64,
    metric_gradient: f64,
    metric_parameters: f64,
    jt_w_j: Array2<f64>,
}

/// Container to perform a curve fit for model, given y and & sy
///
/// The Minimizer is used to initialize and perform a curve fit. For now only 1-dim
/// functions and a Levenberg-Marquardt algorithm is implemented for test purposes.
/// Results have only been verified on simple functions by comparison with
/// an LM implementation from MINPACK.
pub struct Minimizer<'a> {
    pub model: &'a Func1D<'a>,
    pub y: &'a Array1<f64>,
    pub sy: &'a Array1<f64>,
    pub vary_parameter: &'a Array1<bool>,
    pub weighting_matrix: Array1<f64>,
    pub minimizer_parameters: Array1<f64>,
    pub minimizer_y_model: Array1<f64>,
    pub jacobian: Array2<f64>,
    pub parameter_cov_matrix: Array2<f64>,
    pub parameter_errors: Array1<f64>,
    pub lambda: f64,
    pub num_func_evaluation: usize,
    pub max_iterations: usize,
    pub num_varying_params: usize,
    pub num_params: usize,
    pub num_data: usize,
    pub chi2: f64,
    pub dof: usize,
    pub red_chi2: f64,
    pub convergence_message: &'a str,
    pub epsilon1: f64,
    pub epsilon2: f64,
    pub epsilon3: f64,
    pub epsilon4: f64,
    pub lambda_up_fac: f64,
    pub lambda_down_fac: f64,
}

impl<'a> Minimizer<'a> {
    /// Initializes the LM-algorithm. Performs first calculation of model & gradient
    pub fn init<'b>(
        model: &'b Func1D,
        y: &'b Array1<f64>,
        sy: &'b Array1<f64>,
        vary_parameter: &'b Array1<bool>,
        lambda: f64,
    ) -> Minimizer<'b> {
        // at initialization
        let initial_parameters = model.parameters.clone();
        let minimizer_y_model = model.for_parameters(&initial_parameters);

        // calculate number of parameters that are being varied
        let num_varying_params = vary_parameter
            .iter()
            .fold(0, |sum, val| if *val { sum + 1 } else { sum });
        let num_params = initial_parameters.len();
        let num_data = model.domain.len();
        let chi2 = chi2(y, &minimizer_y_model, sy);
        let dof = num_data - num_varying_params;
        let red_chi2 = chi2 / (dof as f64);

        // initialize jacobian
        // J is the parameter gradient of f at the current values
        let j = model.parameter_gradient(&initial_parameters, vary_parameter, &minimizer_y_model);

        // W = 1 / sy^2, only diagonal is considered
        let weighting_matrix: Array1<f64> = sy.map(|x| 1.0 / x.powi(2));

        Minimizer {
            model,
            y,
            sy,
            vary_parameter,
            weighting_matrix,
            minimizer_parameters: initial_parameters,
            minimizer_y_model,
            jacobian: j,
            parameter_cov_matrix: Array2::zeros((num_varying_params, num_varying_params)),
            parameter_errors: Array1::zeros(num_params),
            lambda,
            num_func_evaluation: 0,
            max_iterations: 10 * num_varying_params,
            num_data,
            num_varying_params,
            num_params,
            chi2,
            dof,
            red_chi2,
            convergence_message: "",
            epsilon1: 1e-5,
            epsilon2: 1e-5,
            epsilon3: 1e-5,
            epsilon4: 1e-3,
            lambda_up_fac: 11.0,
            lambda_down_fac: 9.0,
        }
    }

    /// Performs a Levenberg Marquardt step
    ///
    /// determine change to parameters by solving the equation
    /// [J^T W J + lambda diag(J^T W J)] delta = J^T W (y - f)
    /// for delta
    pub fn lm(&mut self) -> MinimizationStep {
        // J^T is cloned to be multiplied by weighting_matrix later
        let mut jt = self.jacobian.clone().reversed_axes();

        // multiply J^T with W to obtain J^T W
        for i in 0..self.num_data {
            let mut col = jt.column_mut(i);
            col *= self.weighting_matrix[i];
        }

        // calculate J^T W (y - f) (rhs of LM step)
        let b = jt.dot(&(self.y - &self.minimizer_y_model));

        // calculate J^T W J + lambda*diag(J^T W J)  [lhs of LM step]
        // first J^T W J
        let jt_w_j = jt.dot(&self.jacobian);

        let lambda_diag_j_t_w_j = self.lambda * &jt_w_j.diag();
        let mut a = jt_w_j.clone();
        for i in 0..self.num_varying_params {
            a[[i, i]] += lambda_diag_j_t_w_j[i];
        }

        // solve LM step for delta
        let delta: Array1<f64> = matrix_solve(&a, &b);

        // create delta with length of total number of parameters
        let mut delta_all: Array1<f64> = Array1::zeros(self.num_params);
        let mut idx_vary_param = 0;
        for i in 0..self.num_params {
            if self.vary_parameter[i] {
                delta_all[i] = delta[idx_vary_param];
                idx_vary_param += 1;
            }
        }

        // calculate metrics to determine convergence
        let mut metric = delta.dot(&b);

        for i in 0..self.num_varying_params {
            metric += delta[i].powi(2) * lambda_diag_j_t_w_j[i];
        }

        // take maximum of the absolute value in the respective arrays as metric for the
        // convergence of either the gradient or the parameters
        let metric_gradient = b.map(|x| x.abs()).to_vec().iter().cloned().fold(f64::NAN, f64::max);

        let metric_parameters = (&delta_all / &self.minimizer_parameters)
            .map(|x| x.abs())
            .to_vec()
            .iter()
            .cloned()
            .fold(f64::NAN, f64::max);

        let updated_parameters = &self.minimizer_parameters + &delta_all;

        let updated_model = self.model.for_parameters(&updated_parameters);
        let updated_chi2 = chi2(self.y, &updated_model, self.sy);
        let red_chi2 = updated_chi2 / (self.dof as f64);

        MinimizationStep {
            parameters: updated_parameters,
            delta,
            y_model: updated_model,
            chi2: updated_chi2,
            red_chi2,
            metric,
            metric_gradient,
            metric_parameters,
            jt_w_j,
        }
    }

    /// Fit routine that performs LM steps until one convergence criteria is met
    ///
    /// Follows the description from <http://people.duke.edu/~hpgavin/ce281/lm.pdf>
    pub fn minimize(&mut self) {
        let mut iterations = 0;
        let inverse_parameter_cov_matrix: Array2<f64>;

        loop {
            let update_step = self.lm();
            iterations += 1;
            info!(">>>>iterations {}", iterations);
            // compare chi2 before and after with respect to metric to decide if step is accepted
            let rho = (self.chi2 - update_step.chi2) / update_step.metric;

            if rho > self.epsilon4 {
                //new parameters are better, update lambda
                self.lambda = (self.lambda / self.lambda_down_fac).max(1e-7);

                // update jacobian
                if iterations % 2 * self.num_varying_params == 0 {
                    // at every 2*n steps update jacobian by explicit calculation
                    // requires #params function evaluations
                    self.jacobian = self.model.parameter_gradient(
                        &self.minimizer_parameters,
                        self.vary_parameter,
                        &self.minimizer_y_model,
                    );
                    self.num_func_evaluation += self.num_varying_params;
                } else {
                    // otherwise update jacobian with Broyden rank-1 update formula
                    let norm_delta = update_step.delta.dot(&update_step.delta);
                    let diff = &update_step.y_model - &self.minimizer_y_model - self.jacobian.dot(&update_step.delta);
                    let mut jacobian_change: Array2<f64> = Array2::zeros((self.num_data, self.num_varying_params));

                    for i in 0..self.num_varying_params {
                        let mut col_slice = jacobian_change.slice_mut(s![.., i]);
                        col_slice.assign(&(&diff * update_step.delta[i] / norm_delta));
                    }

                    self.jacobian = &self.jacobian + &jacobian_change;
                }

                // store new state in Minimizer
                self.minimizer_parameters = update_step.parameters;
                self.minimizer_y_model = update_step.y_model;
                self.chi2 = update_step.chi2;
                self.red_chi2 = update_step.red_chi2;

                // check convergence criteria
                // gradient converged
                if update_step.metric_gradient < self.epsilon1 {
                    self.convergence_message = "Gradient converged";
                    inverse_parameter_cov_matrix = update_step.jt_w_j;
                    break;
                };

                // parameters converged
                if update_step.metric_parameters < self.epsilon2 {
                    self.convergence_message = "Parameters converged";
                    inverse_parameter_cov_matrix = update_step.jt_w_j;
                    break;
                };

                // chi2 converged
                if update_step.red_chi2 < self.epsilon3 {
                    self.convergence_message = "Chi2 converged";
                    inverse_parameter_cov_matrix = update_step.jt_w_j;
                    break;
                };
                if iterations >= self.max_iterations {
                    self.convergence_message = "Reached max. number of iterations";
                    inverse_parameter_cov_matrix = update_step.jt_w_j;
                    break;
                }
            } else {
                // new chi2 not good enough, increasing lambda
                self.lambda = (self.lambda * self.lambda_up_fac).min(1e7);
                // step is rejected, update jacobian by explicit calculation
                self.jacobian = self.model.parameter_gradient(
                    &self.minimizer_parameters,
                    self.vary_parameter,
                    &self.minimizer_y_model,
                );
            }
        }

        // calculate parameter covariance matrix using the LU decomposition
        let (l, u, p) = lu_decomp(&inverse_parameter_cov_matrix);
        for i in 0..self.num_varying_params {
            let mut unit_vector = Array1::zeros(self.num_varying_params);
            unit_vector[i] = 1.0;
            let mut col_slice = self.parameter_cov_matrix.slice_mut(s![.., i]);
            col_slice.assign(&lu_matrix_solve(&l, &u, &p, &unit_vector));
        }
        // parameter fit errors are the sqrt of the diagonal

        let mut idx_vary_param = 0;
        let mut all_errors: Array1<f64> = Array1::zeros(self.num_params);
        for i in 0..self.num_params {
            if self.vary_parameter[i] {
                all_errors[i] = (self.parameter_cov_matrix[[idx_vary_param, idx_vary_param]] * self.red_chi2).sqrt();
                idx_vary_param += 1;
            }
        }
        self.parameter_errors = all_errors;
    }

    /// Prints report of a performed fit
    pub fn report(&self) {
        // calculate coefficient of determination
        let r2 = self.calculate_r2();

        info!("\t #Chi2:\t{:.6}", self.chi2);
        info!("\t #Red. Chi2:\t{:.6}", self.red_chi2);
        info!("\t #R2:\t{:.6}", r2);
        info!("\t #Func. Evaluations:\t{}", self.num_func_evaluation);
        info!("\t #Converged by:\t{}", self.convergence_message);
        info!("---- Parameters ----");
        for i in 0..self.minimizer_parameters.len() {
            if self.vary_parameter[i] {
                info!(
                    "{:.8} +/- {:.8} ({:.2} %)\t(init: {})",
                    self.minimizer_parameters[i],
                    self.parameter_errors[i],
                    (self.parameter_errors[i] / self.minimizer_parameters[i]).abs() * 100.0,
                    self.model.parameters[i]
                );
            } else {
                info!("{:.8}", self.minimizer_parameters[i]);
            }
        }
    }

    /// Calculate the coefficient of determination

    pub fn calculate_r2(&self) -> f64 {
        let mean_y = self.y.sum() / self.y.len() as f64;
        let mut res_sum_sq = 0.0;
        let mut tot_sum_sq = 0.0;
        for i in 0..self.y.len() {
            res_sum_sq += (self.y[i] - self.minimizer_y_model[i]).powi(2);
            tot_sum_sq += (self.y[i] - mean_y).powi(2);
        }
        1.0 - res_sum_sq / tot_sum_sq
    }
}

/// 1.0 / (1.0 + a * x ** (2 * b))
pub fn curve(p: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
    x.map(|&val| 1.0 / (1.0 + p[0] * val.powf(2.0 * p[1])))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn calculate() {
        let spread = 1.0;
        let min_dist = 0.1;
        let p = array![2.0, 1.0];
        let x: Array1<f64> = Array1::range(0.0, 3.0 * spread, spread / 100.0);

        let y: Array1<f64> = x.map(|&x| {
            if x < min_dist {
                1.0
            } else {
                (-(x - min_dist) / spread).exp()
            }
        });
        let vec = y.to_vec();
        println!("{}", vec[0]);
        let model = Func1D::new(&p, &x, curve);
        let sy = Array1::from(vec![1.0; x.len()]);
        let vary_parameter = array![true, true];
        let lambda = 1.0;
        let mut minimizer = Minimizer::init(&model, &y, &sy, &vary_parameter, lambda);
        minimizer.minimize();
        minimizer.report();
    }
}
