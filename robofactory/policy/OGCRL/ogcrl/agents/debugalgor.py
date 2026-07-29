from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from ogcrl.utils.encoders import GCEncoder, encoder_modules
from ogcrl.utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from ogcrl.utils.networks import GCActor, GCDiscreteActor


class DebugAgent(flax.struct.PyTreeNode):
    """A lightweight debug agent that keeps the same input/output interface as the OGCRL actor."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def actor_loss(self, batch, grad_params, rng=None):
        dist = self.network.select("actor")(batch["observations"], batch["actor_goals"], params=grad_params)
        log_prob = dist.log_prob(batch["actions"])
        actor_loss = -log_prob.mean()
        info = {
            "actor_loss": actor_loss,
            "bc_log_prob": log_prob.mean(),
        }
        if not self.config["discrete"]:
            info.update(
                {
                    "mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
                    "std": jnp.mean(dist.scale_diag),
                }
            )
        return actor_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        loss, info = self.actor_loss(batch, grad_params, rng)
        return loss, {f"actor/{k}": v for k, v in info.items()}

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        dist = self.network.select("actor")(observations, goals, temperature=temperature)
        return dist.sample(seed=seed)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config["discrete"]:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        encoders = dict()
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["actor"] = GCEncoder(concat_encoder=encoder_module())

        if config["discrete"]:
            actor_def = GCDiscreteActor(
                hidden_dims=(64, 64),
                action_dim=action_dim,
                gc_encoder=encoders.get("actor"),
            )
        else:
            actor_def = GCActor(
                hidden_dims=(64, 64),
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=True,
                gc_encoder=encoders.get("actor"),
            )

        network_info = dict(
            actor=(actor_def, (ex_observations, ex_goals)),
        )
        network_def = ModuleDict({k: v[0] for k, v in network_info.items()})
        network_tx = optax.adam(learning_rate=config["lr"])
        network_args = {k: v[1] for k, v in network_info.items()}
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="debugalgor",
            lr=3e-4,
            batch_size=16,
            actor_hidden_dims=(64, 64),
            value_hidden_dims=(64, 64),
            layer_norm=False,
            discount=0.99,
            tau=0.005,
            expectile=0.7,
            low_alpha=1.0,
            high_alpha=1.0,
            subgoal_steps=1,
            rep_dim=8,
            low_actor_rep_grad=False,
            const_std=True,
            discrete=False,
            encoder=ml_collections.config_dict.placeholder(str),
            dataset_class="GCDataset",
            value_p_curgoal=0.0,
            value_p_trajgoal=1.0,
            value_p_randomgoal=0.0,
            value_geom_sample=False,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=True,
            p_aug=0.0,
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
    return config