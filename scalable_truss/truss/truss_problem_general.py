import logging
import multiprocessing as mp
import time
import warnings

import numpy as np
from pymoo.core.problem import Problem

from scalable_truss.truss.fea.run_fea import run_fea
from scalable_truss.truss.generate_truss import gen_truss
from innovization.legacy.graph_matrix_innovization import GraphMatrixInnovization
from innovization.vrg_innovization import VRGInnovization
from innovization.interactive.artificial_decision_maker import ArtificialDM
from innovization.interactive.human_decision_maker import HumanDM
from innovization.constants import *
from utils.general import *

logger = logging.getLogger(__name__)


class TrussProblemGeneral(Problem):
    """Generalizes the pymoo truss optimization problem. All node coordinates and beam sizes can be potentially
    considered decision variables."""

    def __init__(self, n_shape_var, shape_var_mode='l', n_cores=mp.cpu_count() // 4,
                 repair_inequality=False, learning_interval=10, repair_interval=1,
                 repair_power=False, symmetry=('xz', 'yz'), decision_maker=None, top_n_rules=-1, **kwargs):
        self.obj_label = ('Weight (kg)', 'Compliance (m/N)')
        # Truss material properties
        self.density = 7121.4  # kg/m3
        self.elastic_modulus = 200e9  # Pa
        self.yield_stress = 248.2e6  # Pa

        # Constraints
        self.max_allowable_displacement = 0.025  # Max displacements of all nodes in x, y, and z directions

        self.force = np.array([0, 0, -5000])

        self.shape_var_mode = shape_var_mode
        self.n_size_var = 0
        self.grouped_size_vars = []
        self.symmetry = symmetry
        if self.symmetry is None:
            self.symmetry = ()

        self.smoothen = True
        if 'disable-smoothening' in kwargs:
            if kwargs['disable-smoothening']:
                self.smoothen = False

        if 'xz' in self.symmetry and 'yz' in self.symmetry:
            # Two planes of symmetry through the middle of truss: x-z plane and y-z plane

            # Set the shape variables
            self.n_shape_var = n_shape_var
            self.coordinates, self.connectivity, self.fixed_nodes, self.load_nodes, self.member_groups \
                = gen_truss(n_shape_nodes=2 * self.n_shape_var - 1)
            self.n_member = self.connectivity.shape[0]
            self.n_node = self.coordinates.shape[0]

            # Set the size variables
            self.grouped_size_vars.append(np.arange(0, len(self.member_groups['straight_x'][0])//2 * 2))
            self.n_size_var += len(self.member_groups['straight_x'][0]) // 2 * 2

            self.grouped_size_vars.append(np.arange(self.n_size_var,
                                                    self.n_size_var + len(
                                                        self.member_groups['straight_xz'][0]) // 2 + 1))
            self.n_size_var += len(self.member_groups['straight_xz'][0]) // 2 + 1

            self.grouped_size_vars.append(np.arange(self.n_size_var,
                                                    self.n_size_var
                                                    + (len(self.member_groups['straight_xy'][0]) // 2 + 1) * 2))
            self.n_size_var += (len(self.member_groups['straight_xy'][0]) // 2 + 1) * 2

            self.grouped_size_vars.append(np.arange(self.n_size_var,
                                                    self.n_size_var + len(self.member_groups['slanted_xz'][0])))
            self.n_size_var += len(self.member_groups['slanted_xz'][0])

            self.grouped_size_vars.append(np.arange(self.n_size_var,
                                                    self.n_size_var + len(self.member_groups['cross_yz_end'][0]) // 2))
            self.n_size_var += len(self.member_groups['cross_yz_end'][0]) // 2

            self.grouped_size_vars.append(np.arange(self.n_size_var,
                                                    self.n_size_var + len(self.member_groups['cross_xy'][0]) // 2 * 2))
            self.n_size_var += len(self.member_groups['cross_xy'][0]) // 2 * 2
        else:
            # No symmetry
            print("No symmetry conditions supplied. Using full truss formulation.")
            # Set shape vars
            self.n_shape_var = n_shape_var
            self.coordinates, self.connectivity, self.fixed_nodes, self.load_nodes, self.member_groups \
                = gen_truss(n_shape_nodes=self.n_shape_var)

            # Set size vars
            self.n_member = self.connectivity.shape[0]
            self.n_node = self.coordinates.shape[0]
            self.grouped_size_vars.append(np.arange(0, self.n_member))
            self.n_size_var = self.n_member

        self.fixed_nodes = self.fixed_nodes.reshape(-1, 1)
        self.load_nodes = self.load_nodes.reshape(-1, 1)

        print(f"No. of shape vars = {self.n_shape_var}")
        print(self.force)

        n_var = self.n_shape_var + self.n_size_var

        # Parallelization
        self.n_cores = n_cores
        if n_cores > mp.cpu_count():
            self.n_cores = mp.cpu_count()

        zmin, zmax = -25, 3.5
        lmin, lmax = 0.5, 29  # l = 4 - z-coord
        if shape_var_mode == 'z':
            # FIXME: Crucial parameters not set
            # Represent truss shape by varying the z coordinates
            xl = np.concatenate((0.005 * np.ones(self.n_size_var), zmin * np.ones(self.n_shape_var)))
            xu = np.concatenate((0.100 * np.ones(self.n_size_var), zmax * np.ones(self.n_shape_var)))
        else:
            # Represent truss shape by through the length (l) of the vertical members. l = 4 - z-coord
            xl = np.concatenate((0.005 * np.ones(self.n_size_var), lmin * np.ones(self.n_shape_var)))
            xu = np.concatenate((0.100 * np.ones(self.n_size_var), lmax * np.ones(self.n_shape_var)))

            # Innovization
            self.percent_pf = 0  # How much of the population consists of non-dominated solutions in the latest generation
            # TODO: rep_operator_selected only works for power laws
            self.rep_operator_selected = None  # Stores the repair operators selected to repair each offspring
            self.learning_interval = learning_interval
            self.repair_interval = repair_interval
            self.repair_power = repair_power
            self.repair_inequality = repair_inequality
            if self.repair_power and self.repair_inequality:
                self.rel_type = POWER_LAW_INEQUALITY_STR
            elif self.repair_power:
                self.rel_type = POWER_LAW_STR
            elif self.repair_inequality:
                self.rel_type = INEQUALITY_RULE_STR
            else:
                warnings.warn(f"Unsupported relation type. Only power law, inequality, or both are supported.")
                self.rel_type = None
        # Ignore pu voltage and transformer tap ratio variables from innovization since their range is too narrow.
        self.ignore_vars = []
        # self.var_groups = np.setdiff1d(np.arange(n_var), self.ignore_vars).reshape(1, -1)
        self.var_groups = []
        n_vars_in_groups = 0
        for g in self.member_groups['straight_x']:
            self.var_groups.append(np.setdiff1d(np.array(g), self.ignore_vars))
            n_vars_in_groups += len(g)
        for g in self.member_groups['straight_xz']:
            self.var_groups.append(np.setdiff1d(np.array(g), self.ignore_vars))
            n_vars_in_groups += len(g)
        # for g in self.member_groups['straight_xy']:
        #     self.var_groups.append(np.setdiff1d(np.array(g), self.ignore_vars))
        #     n_vars_in_groups += len(g)
        self.var_groups.append([g for g in range(n_var - self.n_shape_var, n_var)])
        print(f"{n_vars_in_groups} variables are in groups")
        # self.innov stores the innovization agent. Normalization is used during rule learning.
        if repair_power or repair_inequality:
            self.innov = VRGInnovization(n_var=n_var, groups=self.var_groups, const_tol=0.001,
                                         xl=xl, xu=xu, power_law_normalized=True, rel_type=self.rel_type, **kwargs)
        else:
            self.innov = None
        self.innov_info_available = False  # Flag to indicate whether innovization rules are available to implement

        if 'decision_maker' in kwargs and kwargs['decision_maker'] is not None:
            dm_class, dm_args = get_decision_maker(kwargs['decision_maker'])
            self.decision_maker = dm_class(**dm_args)
        else:
            self.decision_maker = None

        super().__init__(n_var=n_var, n_obj=2, n_constr=2, xl=xl, xu=xu)
        print(f"Number of variables = {self.n_var}")
        print(f"Number of constraints = {self.n_constr}")

    @staticmethod
    def set_conectivity_matrix(connectivity, r, member_groups, symmetry):
        # if symmetry is None or len(symmetry) == 0:
        #     connectivity[:, 2] = r
        #     return connectivity
        if 'xz' in symmetry and 'yz' in symmetry:
            r_indx = 0
            m = member_groups['straight_x']
            connectivity[m[0][:len(m[0]) // 2], 2] = r[r_indx:r_indx + len(m[0]) // 2]  # Bottom
            connectivity[m[0][len(m[0]) // 2:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2])
            connectivity[m[2][:len(m[0]) // 2], 2] = r[r_indx:r_indx + len(m[0]) // 2]  # Bottom
            connectivity[m[2][:len(m[0]) // 2], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2])
            r_indx += len(m[0]) // 2

            connectivity[m[1][:len(m[0]) // 2], 2] = r[r_indx:r_indx + len(m[0]) // 2]  # Bottom
            connectivity[m[1][len(m[0]) // 2:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2])
            connectivity[m[3][:len(m[0]) // 2], 2] = r[r_indx:r_indx + len(m[0]) // 2]  # Bottom
            connectivity[m[3][len(m[0]) // 2:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2])
            r_indx += len(m[0]) // 2

            m = member_groups['straight_xz']
            connectivity[m[0][:len(m[0]) // 2 + 1], 2] = r[r_indx:r_indx + len(m[0]) // 2 + 1]  # y = 0
            connectivity[m[1][:len(m[0]) // 2 + 1], 2] = r[r_indx:r_indx + len(m[0]) // 2 + 1]  # y = 4
            connectivity[m[0][len(m[0]) // 2 + 1:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2 + 1][:-1])
            connectivity[m[1][len(m[0]) // 2 + 1:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2 + 1][:-1])
            r_indx += len(m[0]) // 2 + 1

            m = member_groups['straight_xy']
            connectivity[m[0][:len(m[0]) // 2 + 1], 2] = r[r_indx:r_indx + len(m[0]) // 2 + 1]  # z = 0
            connectivity[m[1][:len(m[0]) // 2 + 1], 2] = r[r_indx:r_indx + len(m[0]) // 2 + 1]  # z = 4
            connectivity[m[0][len(m[0]) // 2 + 1:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2 + 1][:-1])  # z = 0
            connectivity[m[1][len(m[0]) // 2 + 1:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2 + 1][:-1])  # z = 4
            r_indx += len(m[0]) // 2 + 1

            m = member_groups['slanted_xz']
            connectivity[m[0], 2] = r[r_indx:r_indx + len(m[0])]
            connectivity[m[1], 2] = np.flip(r[r_indx:r_indx + len(m[0])])
            connectivity[m[2], 2] = r[r_indx:r_indx + len(m[0])]
            connectivity[m[3], 2] = np.flip(r[r_indx:r_indx + len(m[0])])
            r_indx += len(m[0])

            m = member_groups['cross_yz_end']
            connectivity[m[0], 2] = np.array(r[r_indx], r[r_indx])  # x = 0
            connectivity[m[1], 2] = np.array(r[r_indx], r[r_indx])  # x = 72
            r_indx += 1

            m = member_groups['cross_xy']
            connectivity[m[0][:len(m[0]) // 2], 2] = r[r_indx:r_indx + len(m[0]) // 2]  # z = 0
            connectivity[m[0][len(m[0]) // 2:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2])  # z = 0

            connectivity[m[2][:len(m[0]) // 2], 2] = r[r_indx:r_indx + len(m[0]) // 2]  # z = 0
            connectivity[m[2][len(m[0]) // 2:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2])  # z = 0
            r_indx += len(m[0]) // 2

            connectivity[m[1][:len(m[0]) // 2], 2] = r[r_indx:r_indx + len(m[0]) // 2]  # z = 4
            connectivity[m[1][len(m[0]) // 2:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2])  # z = 4

            connectivity[m[3][:len(m[0]) // 2], 2] = r[r_indx:r_indx + len(m[0]) // 2]  # z = 4
            connectivity[m[3][len(m[0]) // 2:], 2] = np.flip(r[r_indx:r_indx + len(m[0]) // 2])  # z = 4

            return connectivity

        else:
            connectivity[:, 2] = r

            return connectivity

    @staticmethod
    def set_coordinate_matrix(coordinates, shape_var, n_shape_var, shape_var_mode, symmetry):
        # if symmetry is None or len(symmetry) == 0:
        #     coordinates = z
        #
        #     return coordinates
        # Change node coordinates according to the shape decision variables
        if shape_var_mode == 'l':
            # Shape variable expressed as length of vertical members
            z = 4 - shape_var
        else:
            # Shape variable expressed as z-coordinates of bottom node
            z = shape_var
        if 'xz' in symmetry and 'yz' in symmetry:
            coordinates[:n_shape_var, 2] = z
            coordinates[(2 * n_shape_var - 1) * 2:(2 * n_shape_var - 1) * 2 + n_shape_var, 2] = z
            coordinates[n_shape_var:2 * n_shape_var - 1, 2] = np.flip(z[:-1])
            coordinates[(2 * n_shape_var - 1) * 2 + n_shape_var:(2 * n_shape_var - 1) * 2 + 2 * n_shape_var - 1, 2] \
                = np.flip(z[:-1])

        else:
            coordinates[:n_shape_var, 2] = z
            coordinates[2 * n_shape_var:3 * n_shape_var, 2] = z

        # print(coordinates)

        return coordinates

    @staticmethod
    def calc_obj(x_row_indx, x, coordinates, connectivity, member_groups, fixed_nodes, load_nodes, force, density,
                 elastic_modulus, yield_stress, max_allowable_displacement, n_shape_var, shape_var_mode, symmetry,
                 structure_type='truss',):
        r = np.copy(x[:-n_shape_var])  # Radius of each element
        shape_var = np.copy(x[-n_shape_var:])  # Z-coordinate of bottom members

        connectivity = TrussProblemGeneral.set_conectivity_matrix(connectivity=connectivity, r=r,
                                                                  member_groups=member_groups,
                                                                  symmetry=symmetry)

        coordinates = TrussProblemGeneral.set_coordinate_matrix(coordinates=coordinates, shape_var=shape_var,
                                                                n_shape_var=n_shape_var,
                                                                shape_var_mode=shape_var_mode,
                                                                symmetry=symmetry)

        weight, compliance, stress, strain, u, x0_new = run_fea(np.copy(coordinates), np.copy(connectivity),
                                                                fixed_nodes,
                                                                load_nodes, force, density,
                                                                elastic_modulus, structure_type=structure_type)
        del_coord = np.array(x0_new) - coordinates

        f = np.array([weight, compliance])

        # Allow displacement only in -z direction
        if np.max(del_coord[:, 2]) > 0:
            g3 = np.max(del_coord[:, 2])
        else:
            g3 = -1
        g = np.array([np.max(np.abs(stress)) - yield_stress, np.max(np.abs(u)) - max_allowable_displacement, g3])

        return x_row_indx, f, g, stress, strain, u, x0_new, coordinates, connectivity

    def _evaluate(self, x, out, *args, **kwargs):
        x_rep = np.round(x, decimals=2)
        if x_rep.ndim == 1:
            x_rep = x_rep.reshape((1, -1))

        if self.smoothen:
            if self.shape_var_mode == 'z':
                if 'yz' in self.symmetry:
                    x[:, -self.n_shape_var:] = np.flip(np.sort(x[:, -self.n_shape_var:], axis=1), axis=1)
                else:
                    x[:, -self.n_shape_var:-self.n_shape_var // 2] \
                        = np.flip(np.sort(x[:, -self.n_shape_var:-self.n_shape_var // 2], axis=1), axis=1)
                    x[:, -self.n_shape_var // 2:] \
                        = np.sort(x[:, -self.n_shape_var // 2:], axis=1)
            else:
                if 'yz' in self.symmetry:
                    x[:, -self.n_shape_var:] = np.sort(x[:, -self.n_shape_var:], axis=1)
                else:
                    x[:, -self.n_shape_var:-self.n_shape_var // 2] \
                        = np.sort(x[:, -self.n_shape_var:-self.n_shape_var // 2], axis=1)
                    x[:, -self.n_shape_var // 2:] \
                        = np.flip(np.sort(x[:, -self.n_shape_var // 2:], axis=1), axis=1)

        out['rep_time'] = 0
        out['rep_indx'] = np.zeros_like(x)
        out['rep_operator_selected'] = -1 * np.ones(x_rep.shape[0])
        self.rep_operator_selected = out['rep_operator_selected']
        if kwargs['algorithm'].n_gen % self.repair_interval == 0:
            # If power law flag enabled, algo will ignore inequality repair flag. Power law, thus, has a higher
            # priority.
            if self.innov_info_available:
                if self.repair_power:
                    rep_start_time = time.time()
                    percent_to_repair = np.minimum(self.percent_pf, 0.8)
                    n_repair = np.round(percent_to_repair * x_rep.shape[0]).astype(int)
                    print(f"{n_repair}/{x_rep.shape[0]} solutions marked for repair.")
                    sol_to_repair_indx = np.random.choice(np.arange(x_rep.shape[0]), n_repair, replace=False)
                    rep_indx = np.zeros_like(x_rep)
                    rep_operator_selected = np.zeros(x_rep.shape[0])
                    x_r, r_indx, exploration_history, rep_select, vrg_used \
                        = self.innov.repair(x=x_rep[sol_to_repair_indx, :], debug=True)
                    x_rep[sol_to_repair_indx, :], rep_indx[sol_to_repair_indx, :] = x_r, r_indx
                    rep_operator_selected[sol_to_repair_indx] = rep_select
                    rep_time = time.time() - rep_start_time
                    x_rep = np.round(x_rep, decimals=2)
                    if np.any(rep_indx > 0):
                        print(f"repair_power done in {rep_time} sec")
                    out['X'] = x_rep
                    out['rep_indx'] = rep_indx
                    out['rep_time'] = rep_time
                    out['rep_operator_selected'] = rep_operator_selected
                    self.rep_operator_selected = rep_operator_selected
                else:
                    print("Only power law supported")
                # elif self.repair_inequality and len(self.var_groups) > 0:
                #     rep_start_time = time.time()
                #     x_rep, rep_indx = self.innov.repair_inequality(x=x_rep, xl=self.xl, xu=self.xu,
                #                                                    var_groups=self.var_groups,
                #                                                    var_group_score=self.var_group_score,
                #                                                    rel_type=self.rel_type,
                #                                                    percent_pf=self.percent_pf)
                #     rep_time = time.time() - rep_start_time
                #     x_rep = np.round(x_rep, decimals=2)
                #     if np.any(rep_indx > 0):
                #         print(f"repair_inequality done in {rep_time} sec")
                #     out['X'] = x_rep
                #     out['rep_indx'] = rep_indx
                #     out['rep_time'] = rep_time

        f = np.zeros([x.shape[0], self.n_obj])
        g = np.zeros([x.shape[0], self.n_constr])

        if self.n_cores > 1:
            pool = mp.Pool(self.n_cores)
            logging.debug(f"Multiprocessing pool opened. CPU count = {mp.cpu_count()}, Pool Size = {self.n_cores}")
            print(f"Multiprocessing pool opened. CPU count = {mp.cpu_count()}, Pool Size = {self.n_cores}")

            # Call apply_async() for asynchronous evaluation of each population member
            result_objects = [pool.apply_async(TrussProblemGeneral.calc_obj,
                                               args=(i, row, np.copy(self.coordinates),
                                                     np.copy(self.connectivity),
                                                     self.member_groups,
                                                     self.fixed_nodes,
                                                     self.load_nodes, self.force,
                                                     self.density, self.elastic_modulus,
                                                     self.yield_stress,
                                                     self.max_allowable_displacement,
                                                     self.n_shape_var, self.shape_var_mode,
                                                     self.symmetry, 'truss')
                                               )
                              for i, row in enumerate(x)]

            pool.close()  # Need to close the pool to prevent spawning too many processes
            pool.join()
            logging.debug("Parallel objective evaluation complete. Pool closed.")

            # Result_objects is a list of pool.ApplyResult objects
            results = [r.get() for r in result_objects]

            # apply_async() might return results in a different order
            results.sort(key=lambda r: r[0])

            if x.ndim == 1:
                out['X'] = x.flatten()
            else:
                out['X'] = np.copy(x)
            out['F'] = np.array([[r[1][0], r[1][1]] for r in results])
            out['G'] = np.array([[r[2][c] for c in range(self.n_constr)] for r in results])

            out['stress'] = np.array([r[3] for r in results])
            out['strain'] = np.array([r[4] for r in results])
            out['u'] = np.array([r[5] for r in results])
            out['x0_new'] = np.array([r[6] for r in results])
            out['coordinates'] = np.array([r[7] for r in results])
            out['connectivity'] = np.array([r[8] for r in results])
        else:
            print("Sequential execution.")
            result_objects = map(TrussProblemGeneral.calc_obj,
                                 np.arange(x.shape[0]),
                                 x,
                                 [np.copy(self.coordinates) for _ in range(x.shape[0])],
                                 [np.copy(self.connectivity) for _ in range(x.shape[0])],
                                 [self.member_groups for _ in range(x.shape[0])],
                                 [self.fixed_nodes for _ in range(x.shape[0])],
                                 [self.load_nodes for _ in range(x.shape[0])],
                                 [self.force for _ in range(x.shape[0])],
                                 [self.density for _ in range(x.shape[0])],
                                 [self.elastic_modulus for _ in range(x.shape[0])],
                                 [self.yield_stress for _ in range(x.shape[0])],
                                 [self.max_allowable_displacement for _ in range(x.shape[0])],
                                 [self.n_shape_var for _ in range(x.shape[0])],
                                 [self.shape_var_mode for _ in range(x.shape[0])],
                                 [self.symmetry for _ in range(x.shape[0])],
                                 ['truss' for _ in range(x.shape[0])])

            logging.debug("Objective evaluation complete.")

            # Result_objects is a list of pool.ApplyResult objects
            results = list(result_objects)
            # results = result_objects

            if x.ndim == 1:
                out['X'] = x.flatten()
            else:
                out['X'] = np.copy(x)
            out['F'] = np.array([r[1] for r in results])
            out['G'] = np.array([r[2] for r in results])

            out['stress'] = np.array([r[3] for r in results])
            out['strain'] = np.array([r[4] for r in results])
            out['u'] = np.array([r[5] for r in results])
            out['x0_new'] = np.array([r[6] for r in results])
            out['coordinates'] = np.array([r[7] for r in results])
            out['connectivity'] = np.array([r[8] for r in results])
