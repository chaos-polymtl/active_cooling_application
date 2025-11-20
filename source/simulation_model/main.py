from src.time_manager import TimeManager
from src.finite_difference_3d import FiniteDifferenceSolver, FiniteDifferenceSolverSS
from src.adjoint_optimizer import AdjointSS, AdjointTransient

from src.data_manager import DataManager
from src.params import ParametersHandler
from src.pid_controller import PIDController
from src.mpc_controller import MPCController
from src.genetic_mpc import GeneticMPCController

from src.configuration_sweep import ConfigurationSweep

import numpy as np
import sys
import os
from unittest.mock import patch
from contextlib import redirect_stdout

from tqdm import tqdm
import time
import csv

import matplotlib.pyplot as plt

class Main:
    def setup_simulation():
        # Ensure that a path argument is provided
        if len(sys.argv) < 2:
            print("Error: Please provide a path as an argument.")
            sys.exit(1)

        params_path = sys.argv[1]

        # Verify if the path exists
        if not os.path.exists(params_path):
            print(f"Error: The path '{params_path}' does not exist.")
            sys.exit(1)

        # ------------------------------------------------------
        # Construct objects
        # ------------------------------------------------------

        params = ParametersHandler(params_path)
        time_manager = TimeManager(params)
              
        if params.steady:

            finite_difference = FiniteDifferenceSolverSS(params)
        else:
            finite_difference = FiniteDifferenceSolver(params, time_manager)
        data_manager = DataManager(params, finite_difference.points)

        # ------------------------------------------------------
        # Initialize PID controller
        # ------------------------------------------------------
        
        PID = []
        if params.apply_pid_control:
            if params.zone:
                # Apply control to each zone on the actuation face
                for zone in range(len(params.zone_corners)):
                    PID.append(PIDController(params))  # Create PID controller for this zone
            else:
                # Single PID controller for the entire face
                PID = PIDController(params)

        # Save the initial condition
        if params.save_output:
            data_manager.save_vtu(time_manager.current_step, finite_difference.T, finite_difference.heat_flux)

            data_manager.generate_pvd(time_manager.current_step)


        return finite_difference, params, time_manager, data_manager, PID
    
    def transient_simulation(finite_difference, params, time_manager, data_manager, PID):

        log_data = []
        pbar = tqdm(total=params.final_time, desc='Simulation progress')

        if params.apply_echelon:
            echelon_log_data = []

        if params.apply_mpc_control: 
            mpc_log_data = []
            print(f"[MPC] Running in {'INFORMED' if params.mpc_informed else 'UNINFORMED'} mode.")
        
        perturbation_applied = False

        # Log initial state at t=0 (before we solve)
        if params.save_output:
            temperature_at_face = finite_difference.get_temperature_face(params.target_face_id)
            average_temperature_at_face = float(np.mean(temperature_at_face))
            data_manager.save_csv(
                params=params,
                time_manager=time_manager,
                output_path=params.output_path,   
                temperature=average_temperature_at_face                
            )

        while not time_manager.is_finished():

            if (not perturbation_applied) \
                and (getattr(params, "perturbation_type", None) not in (None, "none")) \
                and (time_manager.current_time >= float(params.perturbation_time)):

                if params.perturbation_type == "gaussian_neumann":
                    # Neumann Gaussian (heat flux) -> face must be in params.neumann_boundaries
                    finite_difference.boundary.add_gaussian_neumann(
                        face_id   = int(params.perturbation_face_id),
                        center_xy = params.gaussian_neumann_center,
                        sigma     = float(params.gaussian_neumann_sigma),
                        amplitude = float(params.gaussian_neumann_amplitude),
                        superpose = True
                    )

                elif params.perturbation_type == "gaussian_robin":
                    # Robin Gaussian (spatial h) -> face must be in params.robin_boundaries
                    finite_difference.boundary.add_gaussian_robin(
                        face_id   = int(params.perturbation_face_id),
                        center_xy = params.gaussian_robin_center,
                        sigma     = float(params.gaussian_robin_sigma),
                        h_peak    = float(params.gaussian_robin_h_amplitude),
                        T_inf_jet = float(params.gaussian_robin_T_inf),
                        superpose = True
                    )

                else:
                    print(f"[warn] Unknown perturbation_type={params.perturbation_type!r}; skipping.")

                perturbation_applied = True
                # Optional: small debug print
                print(f"[perturbation] Applied {params.perturbation_type} on face {params.perturbation_face_id} at t={time_manager.current_time:.3f}s")

            if (
                not perturbation_applied 
                and getattr(params, "perturbation_type", None) not in (None, "none")
                and time_manager.current_time >= params.perturbation_time
            ):
                if params.perturbation_type == "gaussian":
                    finite_difference.boundary.add_gaussian_heat_flux(
                        face_id     = params.perturbation_face_id,
                        center_xy   = params.gaussian_center,
                        sigma       = params.gaussian_sigma,    
                        amplitude   = params.gaussian_amplitude,
                        superpose   = True
                    )
                perturbation_applied = True

            # Update the inlet configuration and flow rates with the MPC controller output
            if params.apply_mpc_control:

                # Compute the MPC controller output
                # Update the inlet configuration and flow rates with the MPC output

                ###############################################################################
                # # Genetic Algorithm MPC controller
                
                # MPC = GeneticMPCController(copy.deepcopy(finite_difference),
                #                 target_temperature=params.target_temperature,
                #                 face_id=params.target_face_id)
                # control_output, final_cost, predicted_temperature = MPC.compute_control_action()
                # # finite_difference.boundary.set_inlet_configuration(control_output)

                # target_face = params.target_face_id

                ###############################################################################

                # ################################################################################
                # Standard MPC controller

                ph = getattr(params, "mpc_prediction_horizon", 3)
                ch = getattr(params, "mpc_control_horizon", 1)
                MPC = MPCController(finite_difference, mpc_prediction_horizon=ph, mpc_control_horizon=ch, verbose=getattr(params, "mpc_verbose", False))
                target_face = params.target_face_id
                mpc_target_temperature = params.target_temperature[0]
                informed = getattr(params, "mpc_informed", True)

                control_output, final_cost, predicted_temperature = MPC.compute_mpc_control_action(
                    finite_difference,
                    target_face,
                    mpc_target_temperature,
                    informed
                )

                # ---- Save MPC convergence log for this time step ----
                if getattr(params, "save_output", False) and hasattr(MPC, "iter_costs") and len(MPC.iter_costs) > 0:
                    os.makedirs(params.output_path, exist_ok=True)

                    # CSV with iter, cost, grad_norm
                    log_path = os.path.join(params.output_path, f"mpc_convergence_step_{time_manager.current_step}.csv")
                    with open(log_path, "w") as f:
                        f.write("iter,total_cost,mse,rmse,grad_norm\n")
                        # fall back to NaN if grad_norm_history missing (first run)
                        g_list = getattr(MPC, "grad_norm_history", [])
                        if len(g_list) < len(MPC.iter_costs):
                            g_list = list(g_list) + [float("nan")] * (len(MPC.iter_costs) - len(g_list))
                        for i, (c, m, r, g) in enumerate(zip(MPC.iter_costs,
                                                            getattr(MPC, "iter_mse", [float("nan")] * len(MPC.iter_costs)),
                                                            getattr(MPC, "iter_rmse", [float("nan")] * len(MPC.iter_costs)),
                                                            g_list), start=1):
                            f.write(f"{i},{c},{m},{r},{g}\n")

                    # ----- FIGURE A: LOG(ERROR) VS ITERATION (MSE AND RMSE)
                    iters = range(1, len(MPC.iter_costs)+1)
                    fig = plt.figure()
                    # if hasattr(MPC, "iter_mse") and len(MPC.iter_mse) == len(MPC.iter_costs):
                    #     plt.semilogy(iters, MPC.iter_mse, marker="o", linestyle='-', label="MSE [°C²]")
                    if hasattr(MPC, "iter_rmse") and len(MPC.iter_rmse) == len(MPC.iter_costs):
                        plt.semilogy(iters, MPC.iter_rmse, marker="s", linestyle='--', label="RMSE [°C]")

                    plt.xlabel("Iteration")
                    plt.ylabel("RMSE [°C]")
                    plt.title(f"MPC Convergence (step {time_manager.current_step})")
                    plt.grid(True, alpha=0.4)
                    png_path = os.path.join(params.output_path, f"mpc_convergence_step_{time_manager.current_step}.png")
                    plt.tight_layout()
                    fig.savefig(png_path, dpi=150)
                    plt.close(fig)

                    # ----- FIGURE B: GRAD NORM VS ITERATION
                    if hasattr(MPC, "grad_norm_history") and len(MPC.grad_norm_history) == len(MPC.iter_costs):
                        fig = plt.figure()
                        plt.semilogy(range(1, len(MPC.grad_norm_history)+1), MPC.grad_norm_history, marker="o")
                        plt.xlabel("Iteration")
                        plt.ylabel("‖grad‖₂")
                        plt.title(f"MPC Gradient Norm (step {time_manager.current_step})")
                        plt.grid(True, which="both", alpha=0.4)
                        png_path2 = os.path.join(params.output_path, f"mpc_gradnorm_step_{time_manager.current_step}.png")
                        plt.tight_layout()
                        fig.savefig(png_path2, dpi=150)
                        plt.close(fig)
                # ###########################

                # Log top-face avg temperature at current time (before applying Q0) and predicted avgs
                T_actual = finite_difference.get_temperature_face(target_face)
                T_predicted_avgs = [np.mean(T) for T in predicted_temperature.values()]

                mpc_log_data.append({
                    "time_step": time_manager.current_step,
                    "time": time_manager.current_time,
                    "Q0": control_output.tolist(),
                    "T_avg_top": np.mean(T_actual),
                    "T_predicted_avgs": [float(x) for x in T_predicted_avgs],
                    "Optimized_cost": final_cost
                })

                # Apply the control output to the inlet configuration
                finite_difference.boundary.set_inlet_configuration(control_output)
                ##################################################################################

                print(f"MPC output at t = {time_manager.current_time:.3f}: {control_output} | Cost = {final_cost:.2f}")

            # Update the boundary condition with the PID output
            if params.apply_pid_control:
                # Compute the PID controller output
                # Update the boundary condition with the PID output
                
                if params.zone:
                    target_face = params.zone_faces[1] #above
                    actuation_face = params.zone_faces[0] #below

                    for zone in range(len(params.zone_corners)):
                        target_points = finite_difference.boundary.points_in_zone[target_face][zone]
                        actuation_points = finite_difference.boundary.points_in_zone[actuation_face][zone]

                        T_target_face = np.mean(finite_difference.T[target_points])

                        PID_output, P, I, D = PID[zone].compute_actuator_value(params.target_temperature[zone], T_target_face)

                        for index in actuation_points:
                            finite_difference.boundary.convective_coefficient[index] = PID_output

                else:
                    target_face = params.target_face_id
                    actuation_face = params.actuation_face_id
                    target_face_temperature_avg = np.mean(finite_difference.get_temperature_face(params.target_face_id))
                    PID_output, P, I, D = PID.compute_actuator_value(params.target_temperature[0], target_face_temperature_avg)
                    for index in finite_difference.boundary.dict_boundary_points[actuation_face]:
                        finite_difference.boundary.convective_coefficient[index] = PID_output
                    print(f"PID output: {PID_output}")

            if getattr(params, "apply_echelon", False):
                if time_manager.current_time >= params.echelon_start_time:
                    # Copy the current inlet configuration
                    config = finite_difference.boundary.inlet_configuration.copy()
                    # Apply the step change on the selected jet
                    config[params.echelon_jet_id] = params.echelon_value
                    finite_difference.boundary.set_inlet_configuration(config)
                    print(f"[echelon] Applied step change on jet {params.echelon_jet_id} from {finite_difference.boundary.inlet_configuration[params.echelon_jet_id]} to {params.echelon_value}")

             # --- Echelon step on a single jet (open-loop) ---
            if getattr(params, "apply_echelon", False):
                if time_manager.current_time >= params.echelon_start_time:
                    cfg = finite_difference.boundary.inlet_configuration.copy()
                    cfg[params.echelon_jet_id] = params.echelon_value
                    finite_difference.boundary.set_inlet_configuration(cfg)

            # --- Log (open-loop) current state for echelon figures ---
            if getattr(params, "apply_echelon", False):
                top_face = params.target_face_id
                T_top_arr = finite_difference.get_temperature_face(top_face)
                T_top_avg = float(np.mean(T_top_arr))
                Q_now = np.array(finite_difference.boundary.inlet_configuration, dtype=float)
                echelon_log_data.append({
                    "time_step": time_manager.current_step,
                    "time": time_manager.current_time,
                    "Q": Q_now.tolist(),
                    "T_avg_top": T_top_avg
                })

            # Solve the finite difference model
            finite_difference.solve()

            # fix for open-loop timestamp
            if not (params.apply_mpc_control or params.apply_pid_control):
                time_manager.update_time()

            # Output the current solution to the console in 3D
            if params.save_output:

                # Compute the top-face average temperature for logging
                T_top = finite_difference.get_temperature_face(params.target_face_id)
                T_top_avg = float(np.mean(T_top))

                # Save information related to controllers if they are used
                pid_kwargs = {}
                if params.apply_pid_control:
                    if params.zone:
                        # zone case (log one CSV per zone with that zone’s temperature)
                        for zone in range(len(params.zone_corners)):
                            target_points = finite_difference.boundary.points_in_zone[target_face][zone]
                            T_zone_avg = float(np.mean(finite_difference.T[target_points]))
                            PID_output, P, I, D = PID[zone].compute_actuator_value(params.target_temperature[zone], T_zone_avg)
                            data_manager.save_csv(
                                params=params,
                                time_manager=time_manager,
                                output_path=params.output_path,
                                zone=zone,
                                temperature=T_zone_avg,
                                PID_output=PID_output, P=P, I=I, D=D
                            )
                    else:
                        # non-zoned PID: log top-face average + PID terms
                        pid_kwargs = {"PID_output": PID_output, "P": P, "I": I, "D": D}

                mpc_kwargs = {}
                if params.apply_mpc_control:
                    # Assuming you already computed these this step:
                    #   control_output      -> ndarray of shape (5,) for the 5 actuators
                    #   T_predicted_avgs    -> list of averages over prediction horizon
                    #   final_cost          -> scalar optimizer cost
                    mpc_kwargs = {
                        "Q0": getattr(control_output, "tolist", lambda: control_output)(),
                        "T_predicted_avgs": T_predicted_avgs,
                        "Optimized_cost": final_cost
                    }

                # If not zoned (or even if zoned and you ALSO want a global top log), write the unified row:
                if not params.zone:
                    data_manager.save_csv(
                        params=params,
                        time_manager=time_manager,
                        output_path=params.output_path,   
                        temperature=T_top_avg,
                        **pid_kwargs,
                        **mpc_kwargs
                    )

                # Save as VTU and PVD
                data_manager.save_vtu(time_manager.current_step, finite_difference.T, finite_difference.heat_flux)
                data_manager.generate_pvd(time_manager.current_step)
            
            # Update the time
            if(params.apply_mpc_control or params.apply_pid_control):
                time_manager.update_time()

            pbar.n = np.round(time_manager.current_time, decimals = 3)
            pbar.refresh()

        if params.save_output and params.apply_mpc_control:
            os.makedirs(params.output_path, exist_ok=True)
            with open(os.path.join(params.output_path, "mpc_output_log.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "time_step", "time", "Q0", "T_avg_top", "T_predicted_avgs", "Optimized_cost"
                ])
                writer.writeheader()
                for row in mpc_log_data:
                    writer.writerow(row)
        
        if params.apply_mpc_control and len(mpc_log_data) > 0:
            # Gather series
            times = np.array([row["time"] for row in mpc_log_data], dtype=float)
            T_avg = np.array([row["T_avg_top"] for row in mpc_log_data], dtype=float)
            Q_matrix = np.array([row["Q0"] for row in mpc_log_data], dtype=float) # shape (N_steps, 5)

            dt = params.time_step

            # ---------- Figure 1: Temperature tracking with MPC predictions ----------
            fig1, ax1 = plt.subplots(figsize=(8, 4.5))
            ax1.plot(times, T_avg, label="Measured temperature", linewidth=2)

            # Overlay predicted temps as small dotted segments at future times
            # For each time i, plot predictions at t_i + k*dt (k=1..N_pred)
            for i, row in enumerate(mpc_log_data):
                preds = row["T_predicted_avgs"]  # list of floats (N_pred long)
                if not preds:
                    continue
                # Build x positions for the predictions
                pred_times = times[i] + dt * np.arange(1, len(preds) + 1, dtype=float)
                pred_vals = np.array(preds, dtype=float)
                # Draw as dotted line and markers
                ax1.plot(pred_times, pred_vals, linestyle="--", marker="o", linewidth=1.0, markersize=3,
                        label="_nolegend_")  # don't spam legend

            ax1.set_xlabel("Time [s]")
            ax1.set_ylabel("Temperature [°C]")
            ax1.grid(alpha=0.4)
            ax1.legend(loc="best")
            fig1.tight_layout()

            # Save 
            if params.save_output and params.output_path:
                os.makedirs(params.output_path, exist_ok=True)
                fig1.savefig(os.path.join(params.output_path, "control_temperature.png"), dpi=200)

            # ---------- Figure 2: Flow rates (5 channels) over time ----------
            fig2, ax2 = plt.subplots(figsize=(8, 4.5))
            n_ch = Q_matrix.shape[1] if Q_matrix.ndim == 2 else 0
            ch_labels = [f"Q{i}" for i in range(n_ch)]
            for j in range(n_ch):
                ax2.plot(times, Q_matrix[:, j], label=ch_labels[j], linewidth=1.8)

            ax2.set_xlabel("Time [s]")
            ax2.set_ylabel("Inlet configuration value")
            ax2.grid(alpha=0.4)
            ax2.legend(ncol=min(5, n_ch), loc="best")
            fig2.tight_layout()

            # Save (optional)
            if params.save_output and params.output_path:
                fig2.savefig(os.path.join(params.output_path, "control_flows.png"), dpi=200)

        if params.apply_echelon and len(echelon_log_data) > 0:
            # Gather series
            times = np.array([row["time"] for row in echelon_log_data], dtype=float)
            T_avg = np.array([row["T_avg_top"] for row in echelon_log_data], dtype=float)
            Q_matrix = np.array([row["Q"] for row in echelon_log_data], dtype=float) # shape (N_steps, 5)

            dt = params.time_step

            # ---------- Figure 1: Temperature tracking ----------
            fig1, ax1 = plt.subplots(figsize=(8, 4.5))
            ax1.plot(times, T_avg, linewidth=2)
            ax1.set_xlabel("Time [s]")
            ax1.set_ylabel("Temperature [°C]")
            ax1.grid(alpha=0.4)
            fig1.tight_layout()

            # Save 
            if params.save_output and params.output_path:
                os.makedirs(params.output_path, exist_ok=True)
                fig1.savefig(os.path.join(params.output_path, "open-loop_temperature.png"), dpi=200)

            # ---------- Figure 2: Flow rates (5 channels) over time ----------
            fig2, ax2 = plt.subplots(figsize=(8, 4.5))
            n_ch = Q_matrix.shape[1] if Q_matrix.ndim == 2 else 0
            ch_labels = [f"Q{i}" for i in range(n_ch)]
            step_idx = getattr(params, "echelon_jet_id", None)
            for j in range(n_ch):
                if j == step_idx:
                    ax2.plot(times, Q_matrix[:, j], label=f"Echelon Q{j}", linewidth=2.5, color="C0")
                else:
                    ax2.plot(times, Q_matrix[:, j], label=ch_labels[j], linewidth=1.0, color="0.7", alpha=0.7)

            ax2.set_xlabel("Time [s]")
            ax2.set_ylabel("Flow rate / Inlet configuration value")
            ax2.grid(alpha=0.4)
            ax2.legend(ncol=min(5, n_ch), loc="best")
            fig2.tight_layout()

            # Save (optional)
            if params.save_output and params.output_path:
                fig2.savefig(os.path.join(params.output_path, "open-loop_flows.png"), dpi=200)
            

        pbar.close()
        
        print("Simulation finished.")

    def steady_state_simulation(finite_difference, params, data_manager):

        # Solve the finite difference model
        finite_difference.solve()

        if params.save_output:
            data_manager.save_vtu(1, finite_difference.T, finite_difference.heat_flux)
            data_manager.generate_pvd(current_step = 1)
            np.savetxt(os.path.join(params.output_path, "steady_state_temperature.csv"), finite_difference.get_temperature_face(5), delimiter=",")

        print("Simulation finished.")

    @staticmethod
    def run(test=False):

        finite_difference, params, time_manager, data_manager, PID = Main.setup_simulation()
        
        if params.steady:
            Main.steady_state_simulation(finite_difference, params, data_manager)
        else:
            Main.transient_simulation(finite_difference, params, time_manager, data_manager, PID)
        
        if test:
            return params, finite_difference, PID

    @staticmethod
    def run_application_test(test_file):
        """
        Run the tests while suppressing print statements.

        Parameters:
            test_file (str): Path to the test parameter file.

        Returns:
            tuple: (params, finite_difference, PID) returned by Main.run().
        """
        test_path = os.path.abspath(test_file)
        
        # Suppress stdout during the execution of Main.run()
        with open(os.devnull, 'w') as devnull, redirect_stdout(devnull):
            with patch("sys.argv", ["active-cooling", test_path]):
                params, finite_difference, PID = Main.run(test=True)

        return params, finite_difference, PID
    
    @staticmethod
    def run_parametric_sweep_ss():
        """
        Run a parametric sweep for all the possible permutations of the steady-state simulation.
        """
        start_time = time.time()
       
        finite_difference_ss, params, _, data_manager, _ = Main.setup_simulation()
        
        sweep = ConfigurationSweep(finite_difference_ss, data_manager)
        sweep.get_configs_search_space(flow_rate_increment=params.sweep_granularity)
        sweep.run_configs(run_in_parallel=True, core_count=8)

        end_time = time.time()
        print(f"Parametric sweep setup completed in {end_time - start_time:.2f} seconds.")

    @staticmethod
    def run_adjoint_for_h_reconstruction():
        """
        Run the adjoint optimization to reconstruct the heat transfer coefficient for the steady-state simulation.
        """
        
        finite_difference, params, _, data_manager, _ = Main.setup_simulation()
        
        if params.steady:
            adjoint = AdjointSS(params, finite_difference, data_manager)
        else:
            adjoint = AdjointTransient(params, finite_difference, data_manager)
        
        adjoint.run_nonlinear()


if __name__ == "__main__":
    Main.run()
