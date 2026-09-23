"""Reference and distributed projections with explicit recurrent state."""

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Mesh, Tensor, Topology, tf


@module(entry="step")
class Reference:
    @func
    def step(x: Tensor[(4, 8), "f32"], w: ConstTensor[(8, 6), "f32"], state: Tensor[(4, 6), "f32"]):
        updated = tf.add(tf.matmul(x, w), state)
        return tf.neg(updated), updated


@module(entry="step", topologies=(Topology("gpu", 2),))
class TensorParallel:
    @func
    def step(x: Tensor[(4, 8), "f32"], w: ConstTensor[(8, 6), "f32"], state: Tensor[(4, 6), "f32"]):
        with Mesh(("gpu",), (2,), names=("tp",)) as devices:
            local_x = tf.reshard(x, (4, 8 @ devices.tp), "gmem")
            local_w = tf.reshard(w, (8 @ devices.tp, 6), "gmem")
            partial = tf.matmul(local_x, local_w)
            complete = tf.allreduce(partial, mesh_axis=0)
            replicated_state = tf.reshard(state, (4, 6), "gmem")
            updated = tf.add(complete, replicated_state)
            return tf.neg(updated), updated


@module(entry="step", topologies=(Topology("gpu", 2),))
class ReduceScatterProjection:
    @func
    def step(x: Tensor[(4, 8), "f32"], w: ConstTensor[(8, 6), "f32"], state: Tensor[(4, 6), "f32"]):
        with Mesh(("gpu",), (2,), names=("tp",)) as devices:
            local_x = tf.reshard(x, (4, 8 @ devices.tp), "gmem")
            local_w = tf.reshard(w, (8 @ devices.tp, 6), "gmem")
            partial = tf.matmul(local_x, local_w)
            complete = tf.reducescatter(partial, mesh_axis=0, tensor_axis=0)
            local_state = tf.reshard(state, (4 @ devices.tp, 6), "gmem")
            updated = tf.add(complete, local_state)
            return tf.neg(updated), updated
