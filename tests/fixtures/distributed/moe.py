"""A nonlinear MoE expressed as reference HIR and bounded expert parallel HIR."""

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Mesh, Tensor, Topology, tf

D, N, H, F, E, K = 2, 8, 4, 6, 4, 3
CAPACITY = N * K


@module(entry="run")
class Reference:
    @func
    def run(
        x: Tensor[(D, N, H), "f32"],
        indices: Tensor[(D, N, K), "i64"],
        weights: Tensor[(D, N, K), "f32"],
        w1: ConstTensor[(E, H, F), "f32"],
        w2: ConstTensor[(E, F, H), "f32"],
    ):
        result = tf.full_like(x, value=0.0)
        zeros = tf.full_like(weights, value=0.0)
        for expert in range(E):
            up = tf.slice(w1, (expert, 0, 0), sizes=(1, H, F), strides=(1, 1, 1))
            down = tf.slice(w2, (expert, 0, 0), sizes=(1, F, H), strides=(1, 1, 1))
            expert_result = tf.matmul(tf.relu(tf.matmul(x, up)), down)
            selected = tf.where(tf.cmp_eq(indices, expert), weights, zeros)
            scale = tf.reduce(selected, axes=(-1,), keepdim=True, kind="sum")
            result = tf.add(result, tf.mul(expert_result, scale))
        return result


@module(entry="run", topologies=(Topology("gpu", 4),))
class ExpertParallel:
    @func
    def dispatch(
        x: Tensor[(D, N, H), "f32"],
        indices: Tensor[(D, N, K), "i64"],
        weights: Tensor[(D, N, K), "f32"],
    ):
        with Mesh(("gpu",), (2, 2), names=("dp", "ep")) as devices:
            source = tf.reshard(x, (D @ devices.dp, N @ devices.ep, H), "gmem")
            routes = tf.reshard(indices, (D @ devices.dp, N @ devices.ep, K), "gmem")
            scales = tf.reshard(weights, (D @ devices.dp, N @ devices.ep, K), "gmem")
            return tf.alltoall_dispatch(
                source,
                routes,
                scales,
                num_experts=E,
                capacity=CAPACITY,
                mesh_axis=1,
            )

    @func
    def run(
        x: Tensor[(D, N, H), "f32"],
        indices: Tensor[(D, N, K), "i64"],
        weights: Tensor[(D, N, K), "f32"],
        w1: ConstTensor[(E, H, F), "f32"],
        w2: ConstTensor[(E, F, H), "f32"],
    ):
        packed, scales, source_indices, counts = dispatch(x, indices, weights)  # noqa: F821
        with Mesh(("gpu",), (2, 2), names=("dp", "ep")) as devices:
            source = tf.reshard(x, (D @ devices.dp, N @ devices.ep, H), "gmem")
            up = tf.reshard(w1, (E @ devices.ep, H, F), "gmem")
            down = tf.reshard(w2, (E @ devices.ep, F, H), "gmem")
            expert_result = tf.matmul(tf.relu(tf.matmul(packed, up)), down)
            weighted = tf.mul(expert_result, scales)
            return tf.alltoall_combine(weighted, source_indices, counts, source, mesh_axis=1)
