"""
Minimum 2-frame Joint BA with POSLV Prior.
Optimizes:
  - Rig Extrinsic (LiDAR-to-Camera)
  - Ego-motion corrections (delta 
import numpy as np
import pyceres
import torch
from scipy.spatial.transform import Rotation as R
import ba_singleframe as bas

class POSLVPriorCost(pyceres.CostFunction):
    """Constraint on Ego-pose based on POSLV uncertainty."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    def __init__(self, pose_prior, covariance):
        super().__init__()
        self.prior = pose_g.inv(covariance)).T
        self.set_num_residuals(6)
        self.set_parameter_block_sizes([6])

    def Evaluate(self, params, residuals, jacobians):
        theta = params[0]
        residuals[:] = self.sqrt_info @ (theta - self.prior)
        if jacobians is 
class JointCalibCost(pyceres.CostFunction):
    """
    Residual = π(T_rig^-1 * T_ego^-1 * P_world) - (uv_ref + d)
    Optimizes boi,n

        self.uv_ref = uv_ref
        self.d = d
        self.L = sqrt_info
        self.K = K
        self.N = P_world.shape[0]
        self.set_num_residuals(2 * self.N)
        self.set_parameter_block_sizes([6, 6]) # [Ego Pose, Rig Extrinsic]
  theta_rig[3:6]
        
        # P_cam = R_rig^T * (R_ego^T * (P_w - t_ego) - t_rig)
        P_v = (R_ego.T @ (P - t_ego).T).T
        P_c = (R_rig.T @ (P_v - t_rig).T).T
        return P_c

    def Evaluate(self, params, residuals, jacobians):
        theta_ego, theta_rig = params[0], params[1]
        P_c = self._transform(self.P_w, theta_ego, theta_rig)
        
        fx, fy = self.K[0,0], self.K[1,1]
        cx, cy = self.K[0,2], self.K[1,2]
        
        z = P_c[:, 2]
        proj = np.stack([fx * P_c[:,0]/z + cx, fy * P_c[:,1]/z + cy], axis=1)
        r_raw = proj - self.uv_ref - self.d
        residuals[:] = np.einsum('nij,nj->ni', self.L, r_raw).ravel()
        
        if jacobians is not None:
            # Numerical Jacobian for the joint params
            eps = 1e-6
       b in rang        p_plus = params[b].copy(); p_plus[k] += eps
                        p_minus = params[b].copy(); p_minus[k] -= eps
                        
                        # Re-evaluate with perturbed param
                        def get_res(p, block_idx):
                            p_pair = [params[0], params[1]]
                            p_pair[block_idx] = p
                            pc = self._transform(self.P_w, p_pair[0], p_pair[1])
                            z_ = pc[:, 2]
                            pr = np.stack([fx * pc[:,0]/z_ + cx, fy * pc[:,1]/z_ + cy], axis=1)
                            return np.einsum('nij,nj->ni', self.L, pr - self.uv_ref - self.d).ravel()
                        
                        J[:, k] = (get_res(p_plus, b) - get_res(p_minus, b)) / (2*eps)
                    jacobians[b][:] = J.ravel()
        return True

def solve_2frame_joint(frame_obs_list, rig_prior, poslv_priors, K):
    """
    frame_obs_list: list of dicts from run_inference
    rig_prior: [y, p, r, tx, ty, tz]
    poslv_priors: list of [y, p, r, x, y, z]
    """
    prob = pyceres.Problem()
    theta_rig = np.array(rig_prior, copy=True)
    theta_poses = [np.array(p, copy=True) for p in poslv_priors]

    # POSLV accuracy: 0.01m, 0.01deg approx
    poslv_cov = np.diag([1e-4, 1e-4, 1e-4, 0.01**2, 0.01**2, 0.01**2])

    for i, obs in enumerate(frame_obs_list):
        # 1. POSLV Prior
        prob.add_residual_block(POSLVPriorCost(poslv_priors[i], poslv_cov), None, [theta_poses[i]])
        
        # 2. CalibNet Observations
        P_w = obs['P_wo obs['i
        L = np.zeros_like(Σ)
        fo(add_residual_block(
            JointCalibCost(P_w, obs['uv_ref'], obs['d'], L, K),
            None, [theta_poses[i], theta_rig]
        )
yp.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    options.max_num_iterations = 100 summary)
    

_name= "__maP= np.diag([1e-5]*6) 