###########################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
###########################################################################
"""Provenance-producing geometry transformations for scan points."""

import numpy as np
from aiida import orm
from aiida.engine import calcfunction


@calcfunction
def scale_structure(structure: orm.StructureData, scale_factor: orm.Float) -> orm.StructureData:
    """Scale the structure with the given scaling factor."""
    ase = structure.get_ase().copy()
    ase.set_cell(ase.get_cell() * float(scale_factor) ** (1 / 3), scale_atoms=True)
    return orm.StructureData(ase=ase)


@calcfunction
def set_distance(molecule: orm.StructureData, distance: orm.Float) -> orm.StructureData:
    """
    Move both sites of the molecule to symmetric points around the origin which are on the line connecting the original
    sites and which are separated by the target distance.
    """
    vector_diff = np.array(molecule.sites[1].position) - np.array(molecule.sites[0].position)
    versor_diff = vector_diff / np.linalg.norm(vector_diff)
    new_molecule = molecule.clone()
    new_position = (distance.value * versor_diff) / 2
    new_molecule.reset_sites_positions([-new_position, new_position])
    return new_molecule
