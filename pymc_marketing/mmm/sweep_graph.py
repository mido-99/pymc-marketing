#   Copyright 2022 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""The compiled sweep graph.

The sweep grid is a symbolic input, so the graph's structure depends on the
response graph, the input it replaces and the sweep type, not on the grid's
values.  :func:`compose_sweep` builds the symbolic inputs and graph, and
:class:`~pymc_marketing.mmm.sensitivity_analysis.SensitivityAnalysis` compiles
them for one call.
"""

from __future__ import annotations

import pytensor.tensor as pt
import pytensor.xtensor as ptx
from pytensor.graph.basic import Variable
from pytensor.tensor import TensorVariable
from pytensor.xtensor.type import XTensorVariable, as_xtensor
from pytensor.xtensor.vectorization import vectorize_graph

SWEEP_DIM = "sweep"
"""Name of the dimension the sweep grid is vectorized over."""

SWEEP_NAME = "sweep"
"""Name of the symbolic sweep input every compiled sweep takes."""

MASK_NAME = "mask"
"""Name of the symbolic response-mask input a masked sweep takes."""


def apply_response_mask(
    response: XTensorVariable, mask: TensorVariable
) -> XTensorVariable:
    """Zero the response where a symbolic boolean mask is false.

    The mask is broadcast over the whole response graph, its sample batch
    included, because a ``set_subtensor`` with slices cannot be vectorized.
    Broadcasting the mask rather than the response keeps the mask a plain
    function argument of the compiled sweep.

    Parameters
    ----------
    response : XTensorVariable
        The extracted response graph, with a leading ``sample`` dimension.
    mask : TensorVariable
        Boolean symbolic input with the response's non-sample shape.

    Returns
    -------
    XTensorVariable
        The response with every masked-out element set to zero.
    """
    mask_bcast, _ = pt.broadcast_arrays(mask, response.values)
    return as_xtensor(response.values[~mask_bcast].set(0), dims=response.dims)


def build_sweep_response(
    base_value: XTensorVariable,
    response: XTensorVariable,
    sweep: XTensorVariable,
    sweep_type: str,
) -> XTensorVariable:
    """Compose the sweep graph: input combination, vectorization and date sum.

    Parameters
    ----------
    base_value : XTensorVariable
        The input the sweep replaces: the source model's live input.
    response : XTensorVariable
        The extracted response graph, already masked when a mask is used.
    sweep : XTensorVariable
        The symbolic sweep grid.
    sweep_type : str
        One of ``"multiplicative"``, ``"additive"`` or ``"absolute"``.

    Returns
    -------
    XTensorVariable
        The response summed over ``date``, transposed to ``sample, sweep, ...``.

    Raises
    ------
    ValueError
        If ``sweep_type`` is not one of the three supported types.
    """
    match sweep_type:
        case "multiplicative":
            sweep_combinations = sweep * base_value
        case "additive":
            sweep_combinations = sweep + base_value
        case "absolute":
            sweep_combinations = sweep.broadcast_like(base_value)
        case _:
            raise ValueError(f"Unknown sweep_type {sweep_type!r}")

    sweep_response = vectorize_graph(response, replace={base_value: sweep_combinations})
    return sweep_response.sum(dim="date").transpose("sample", SWEEP_DIM, ...)


def compose_sweep(
    *,
    response: XTensorVariable,
    base_value: XTensorVariable,
    sweep_type: str,
    has_mask: bool,
    mask_shape: tuple[int, ...] | None = None,
) -> tuple[list[Variable], XTensorVariable]:
    """Compose the symbolic inputs and graph of one compiled sweep.

    The response graph and the input it replaces come from the source model, so
    the sweep a call compiles reads that model's live data.

    Parameters
    ----------
    response : XTensorVariable
        The extracted response graph, with a leading ``sample`` dimension.
    base_value : XTensorVariable
        The input the sweep replaces.
    sweep_type : str
        One of ``"multiplicative"``, ``"additive"`` or ``"absolute"``.
    has_mask : bool
        Whether the graph takes a symbolic response mask.
    mask_shape : tuple of int, optional
        Shape of the mask input, required when ``has_mask`` is true.

    Returns
    -------
    list of Variable
        The compiled function's inputs: the sweep grid and, with a mask, the
        mask.
    XTensorVariable
        The sweep graph.

    Raises
    ------
    ValueError
        If ``has_mask`` is true and ``mask_shape`` is not given.
    """
    if has_mask and mask_shape is None:
        raise ValueError("mask_shape is required when has_mask is true.")
    sweep = ptx.xtensor(SWEEP_NAME, dims=(SWEEP_DIM,))
    inputs: list[Variable] = [sweep]
    if has_mask:
        mask = pt.tensor(MASK_NAME, shape=mask_shape, dtype=bool)
        response = apply_response_mask(response, mask)
        inputs.append(mask)
    return inputs, build_sweep_response(base_value, response, sweep, sweep_type)
